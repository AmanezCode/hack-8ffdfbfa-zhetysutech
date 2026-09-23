"""Streamlit demo: the forecast agent on the February 2026 replay."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ARTIFACTS, FORECASTS, ISSUE_HOUR_UTC, SCADA_UTC_OFFSET_HOURS, TEST_END, TEST_START, TURBINES
from src.forecast_agent import ForecastAgent
from src.llm_agent import OperatorAgent

LOGGERS = [logging.getLogger("src.forecast_agent"), logging.getLogger("src.llm_agent")]

# Reference data-viz palette: categorical slots in fixed order, one-hue sequential ramp.
PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "muted": "#898781", "grid": "#2c2c2a"},
}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODEL_NAMES = {
    "physics": "Кривая мощности (физика)",
    "residual": "Кривая + LightGBM на остатках",
    "direct": "LightGBM (прямой)",
    "climatology": "Климатология (среднее по часу)",
    "persistence": "Персистентность*",
}

LLM_LABELS = {
    "auto": "LLM: автовыбор по ключу",
    "openai": "OpenAI (GPT)",
    "claude": "Claude",
    "template": "Без LLM (шаблон)",
    "off": "Не готовить",
}

FLAG_TEXT = {
    "zero_wind_power": lambda f: f"{f['hours']} ч: прогноз ветра 0 м/с, но мощность выше 5% — среднее за час рядом со штилем.",
    "curve_deviation": lambda f: f"{f['hours']} ч: прогноз отклоняется от кривой мощности сильнее, чем в 99% случаев "
                                 f"на проверке ({f['limit']:.0%}) — стоит проверить вручную.",
    "older_weather_run": lambda f: f"{f['hours']} ч: свежего запуска погоды нет в архиве, агент взял более ранний допустимый.",
}


def theme() -> str:
    try:
        return st.context.theme.type or "dark"
    except AttributeError:
        return "dark"


def colors() -> dict:
    return PALETTE[theme()]


def issue_time_for(forecast_date: date) -> pd.Timestamp:
    """Official issue: 23:00 SCADA the evening before, i.e. 17:00 UTC."""
    return (pd.Timestamp(forecast_date) - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC)).tz_localize("UTC")


def to_scada(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_localize(None) + pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def scada_label(iso: str) -> str:
    return to_scada([iso])[0].strftime("%d.%m %H:%M")


@st.cache_data
def load_manifest(turbine_id: int) -> dict:
    return json.loads((ARTIFACTS / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_efficiency() -> dict:
    path = ARTIFACTS / "metrics_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def session_output_dir():
    if "session_dir" not in st.session_state:
        st.session_state["session_dir"] = FORECASTS / "streamlit" / uuid.uuid4().hex[:12]
    return st.session_state["session_dir"]


class UILog(logging.Handler):
    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        self.callback(record.getMessage())


def base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    grid = colors()["grid"]
    fig.update_layout(title=title, height=height, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(title="Время SCADA (UTC+6)", gridcolor=grid)
    fig.update_yaxes(title="Мощность, % от номинальной", range=[0, 100], ticksuffix="%", gridcolor=grid)
    return fig


# ---------------------------------------------------------------- sections

def render_quality(turbine_id: int) -> None:
    manifest = load_manifest(turbine_id)
    chosen = manifest["variant"]
    cv = manifest["cv_mean"]
    folds = manifest["cv_folds"]
    by_lead = {label: np.mean([f["mae_by_lead"][label][chosen] for f in folds]) for label in ("1-24h", "25-48h")}
    efficiency = load_efficiency().get(str(turbine_id), {})

    gain = manifest.get("update_gain", {}).get("+18h", {}).get("improvement")
    interval = manifest.get("interval") or {}

    st.subheader("Качество модели")
    cols = st.columns(5)
    cols[0].metric("Точность (1 − nMAE)", f"{cv[chosen]['accuracy']:.1%}",
                   help="Средняя абсолютная ошибка как доля номинальной мощности, вычтенная из 100%. "
                        "Скользящая проверка: ноябрь 2025 — январь 2026, официальные выпуски 23:00.")
    cols[1].metric("Часов в пределах ±10%", f"{cv[chosen]['within_10pct']:.0%}",
                   help="Доля часов, где прогноз отличается от факта не больше чем на 10% номинала.")
    cols[2].metric("Лучше климатологии", f"{1 - cv[chosen]['MAE'] / cv['climatology']['MAE']:.0%}",
                   help="Снижение MAE относительно среднего профиля мощности по часу суток.")
    if gain is not None:
        cols[3].metric("Пересчёт через 18 ч", f"−{gain:.1%}",
                       help="Насколько меньше ошибка (MAE) прогноза на те же часы, если агент пересчитал его "
                            "с более свежей погодой.")
    if efficiency:
        cols[4].metric("Прогноз на 48 ч", f"{efficiency['agent_cycle_ms_per_forecast']:.0f} мс",
                       help=f"Полный цикл агента; инференс модели {efficiency['inference_ms_per_48h_forecast']:.0f} мс, "
                            f"обучение {efficiency['train_seconds']:.0f} с.")

    st.caption(f"Горизонт 1–24 ч: nMAE {by_lead['1-24h']:.1%} · 25–48 ч: nMAE {by_lead['25-48h']:.1%} · "
               f"лучше кривой мощности на {1 - cv[chosen]['MAE'] / cv['physics']['MAE']:.1%} · "
               f"ошибка суточной энергии {cv[chosen]['daily_energy_error']:.1%}"
               + (f" · 80%-ный интервал покрыл {interval['cv_coverage']:.0%} фактов" if interval.get("cv_coverage") else "")
               + f" · {MODEL_NAMES[chosen]}, версия {manifest['model_version']}")

    with st.expander("Сравнение моделей и baseline"):
        table = pd.DataFrame([
            {"Модель": MODEL_NAMES[name] + (" — выбрана" if name == chosen else ""),
             "Точность, %": 100 * cv[name]["accuracy"], "nMAE, %": 100 * cv[name]["MAE"], "nRMSE, %": 100 * cv[name]["RMSE"]}
            for name in ("physics", "residual", "direct", "climatology", "persistence")
        ])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in table.columns[1:]})
        st.caption("*Персистентность требует свежую SCADA на момент выпуска; в закрытом февральском тесте её нет, "
                   "показана для масштаба. Февральский факт организаторы не публиковали, поэтому точность — по backtest.")


def forecast_figure(frame: pd.DataFrame, analysis: dict, title: str) -> go.Figure:
    c = colors()
    x = to_scada(frame["time"])
    power = 100 * frame["prediction"].to_numpy()
    fig = go.Figure()
    interval = analysis.get("interval_80")
    band = analysis.get("expected_abs_error")
    if interval:
        low, high, label = 100 * np.asarray(interval["low"]), 100 * np.asarray(interval["high"]), "80%-ный интервал"
    elif band:
        error = 100 * np.asarray(band)
        low, high, label = np.clip(power - error, 0, 100), np.clip(power + error, 0, 100), "Ожидаемая ошибка (CV)"
    if interval or band:
        fig.add_trace(go.Scatter(x=x, y=high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=low, line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,0.18)",
                                 name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=100 * frame["level1"], name="Кривая мощности", mode="lines",
                             line=dict(color=c["muted"], width=2, dash="dot"),
                             hovertemplate="%{y:.0f}%<extra>кривая</extra>"))
    fig.add_trace(go.Scatter(x=x, y=power, name="Прогноз агента", mode="lines+markers",
                             line=dict(color=c["series"][0], width=2), marker=dict(size=8),
                             hovertemplate="%{y:.0f}%<extra>прогноз</extra>"))
    return base_layout(fig, title)


def versions_figure(versions: list[tuple[str, pd.DataFrame]]) -> go.Figure:
    series = colors()["series"]
    fig = go.Figure()
    for slot, (label, frame) in enumerate(versions[: len(series)]):
        fig.add_trace(go.Scatter(x=to_scada(frame["time"]), y=100 * frame["prediction"], name=label, mode="lines",
                                 line=dict(color=series[slot], width=2), hovertemplate="%{y:.0f}%"))
    return base_layout(fig, "Версии прогноза по мере поступления свежей погоды")


def render_analysis(analysis: dict) -> None:
    st.markdown("**Анализ результата**")
    days = [d for d in analysis["days"] if d["hours"] >= 12]
    cols = st.columns(max(len(days), 1))
    for col, day in zip(cols, days):
        col.metric(f"{pd.Timestamp(day['date_scada']):%d.%m} · КИУМ", f"{day['capacity_factor']:.0%}",
                   help="Коэффициент использования установленной мощности за сутки по прогнозу.")
        col.caption(f"{day['full_load_hours']:.1f} ч полной мощности · пик {day['peak_power']:.0%} "
                    f"в {day['peak_scada'][-5:]}" + (f" · ошибка ±{day['expected_mae']:.0%}" if day["expected_mae"] else ""))
    if analysis["ramps"]:
        ramps = pd.DataFrame([{"Начало (SCADA)": r["start_scada"], "Направление": "рост" if r["direction"] == "up" else "спад",
                               "Изменение за 3 ч": f"{r['change']:+.0%}"} for r in analysis["ramps"]])
        st.markdown("Рампы (изменение ≥ 30% номинала за 3 ч) — важны для диспетчера:")
        st.dataframe(ramps, hide_index=True, width="stretch")
    revision = analysis.get("revision")
    if revision:
        st.caption(f"Ревизия относительно предыдущего прогноза: {revision['overlap_hours']} общих часов, "
                   f"в среднем {revision['mean_abs_change']:.1%}, максимум {revision['max_abs_change']:.0%} "
                   f"в {revision['max_change_scada'][-11:]}.")
    for flag in analysis["flags"]:
        st.info(FLAG_TEXT[flag["code"]](flag), icon="ℹ️")


def render_briefing(briefing) -> None:
    st.subheader("Брифинг диспетчеру")
    if briefing.provider == "template":
        source = "шаблон без LLM" + (f" — {briefing.note}" if briefing.note else " (ключ LLM не задан)")
    else:
        source = f"{'OpenAI' if briefing.provider == 'openai' else 'Claude'} · {briefing.model}"
    tools = ", ".join(t["tool"] for t in briefing.trace)
    st.caption(f"Написал: {source} · вызвано инструментов агента: {len(briefing.trace)} ({tools}). "
               "Все числа взяты из результатов инструментов.")
    with st.container(border=True):
        st.markdown(briefing.text)


def render_updates(decisions: list[dict]) -> None:
    rows = []
    for d in decisions:
        runs = ", ".join(f"{k.replace('->', '→')} ×{v}" for k, v in (d.get("fresher_runs") or {}).items())
        revision = (d.get("analysis") or {}).get("revision") if d.get("recomputed") else None
        rows.append({
            "Проверка (SCADA)": scada_label(d["check_time"]),
            "Что изменилось": "плановый выпуск" if d.get("trigger") == "scheduled"
            else (f"{d['changed_hours']} из {d['overlap_hours']} ч: {runs}" if d["changed_hours"] else "ничего"),
            "Решение агента": "выпущен по расписанию" if d.get("trigger") == "scheduled"
            else ("пересчитан" if d["recomputed"] else "оставлен без изменений"),
            "Средняя ревизия": f"{revision['mean_abs_change']:.1%}" if revision and d.get("trigger") != "scheduled" else "—",
            "Макс. ревизия": f"{revision['max_abs_change']:.0%}" if revision and d.get("trigger") != "scheduled" else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def heatmap_figure(frame: pd.DataFrame) -> go.Figure:
    pivot = frame.pivot(index="forecast_date", columns="lead_hours", values="prediction").sort_index()
    scale = BLUE_RAMP if theme() == "light" else BLUE_RAMP[::-1]
    fig = go.Figure(go.Heatmap(z=100 * pivot.to_numpy(), x=pivot.columns, y=[f"{d:%d.%m}" for d in pivot.index],
                               colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)], zmin=0, zmax=100,
                               colorbar=dict(title="%", ticksuffix="%"),
                               hovertemplate="прогноз на %{y}<br>час %{x}<br>%{z:.0f}%<extra></extra>"))
    fig.update_layout(title="28 выпусков × 48 часов: прогноз мощности", height=560, margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(title="Час горизонта (1–48)")
    # "01.02" would otherwise be read as the number 1.02.
    fig.update_yaxes(title="Дата прогноза (сутки D+1)", type="category", autorange="reversed")
    return fig


def read_version(agent: ForecastAgent, run_id: str) -> pd.DataFrame:
    path = next(agent.output_dir.glob(f"*_{run_id}.json"))
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["forecast"])


# --------------------------------------------------------------------- page

st.set_page_config(page_title="Wind Forecast Agent", page_icon="🌬️", layout="wide")
st.title("Wind Forecast Agent")
st.caption("Почасовой прогноз выработки ВЭС на 24–48 часов · replay февраля 2026 на архивных прогнозах погоды, "
           "доступных в момент выпуска · время SCADA (UTC+6)")

with st.sidebar:
    st.header("Параметры")
    turbine_id = st.selectbox("Турбина", options=list(TURBINES), format_func=lambda t: f"Турбина {t}")
    mode = st.radio("Режим", ["Один выпуск", "Весь февраль", "Сейчас (live)"])
    forecast_date = None
    with_updates = live = False
    if mode == "Один выпуск":
        forecast_date = st.date_input("Прогноз на сутки", value=date.fromisoformat(TEST_START),
                                      min_value=date.fromisoformat(TEST_START), max_value=date.fromisoformat(TEST_END))
        with_updates = st.toggle("Пересчёт при свежей погоде (каждые 6 ч)", value=True,
                                 help="Агент проверяет, стали ли доступны более свежие запуски погодной модели, "
                                      "и пересчитывает прогноз, только если входные данные изменились.")
    if mode == "Сейчас (live)":
        live = True
        st.caption("Прогноз от текущего часа по свежей погоде Open-Meteo.")
    else:
        live = st.toggle("Живой запрос к Open-Meteo", value=False,
                         help="Кроме закреплённого архива, агент запрашивает погоду по координатам турбины через API.")
    llm = "template"
    if mode != "Весь февраль":
        llm = st.selectbox("Брифинг диспетчеру", options=list(LLM_LABELS), format_func=LLM_LABELS.get,
                           help="LLM сама вызывает инструменты агента и пишет брифинг; ключ берётся из "
                                "OPENAI_API_KEY / ANTHROPIC_API_KEY. Без ключа — шаблон на тех же инструментах.")
    run_clicked = st.button("Запустить агента", type="primary", width="stretch")

render_quality(turbine_id)
st.divider()

request_key = (turbine_id, mode, forecast_date, with_updates, live, llm)
if run_clicked:
    lines: list[str] = []
    panel = st.empty()

    def log(line: str) -> None:
        lines.append(line)
        panel.code("\n".join(lines[-40:]), language="text")

    handler = UILog(log)
    saved = [(logger, logger.level, logger.propagate) for logger in LOGGERS]
    for logger in LOGGERS:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
    cfg = {t: (c["lat"], c["lon"]) for t, c in TURBINES.items()}
    agent = ForecastAgent(output_dir=session_output_dir(), coordinates=cfg)
    state = {"key": request_key}
    try:
        with st.spinner("Агент работает…"):
            if mode == "Сейчас (live)":
                issue = pd.Timestamp.now(tz="UTC").floor("h")
                result = agent.run(turbine_id, issue, refresh_weather=True)
                state["result"], state["analysis"], state["issue"] = result, result.attrs["agent"]["analysis"], issue
            elif mode == "Один выпуск":
                issue = issue_time_for(forecast_date)
                state["issue"] = issue
                if with_updates:
                    decisions = agent.run_update_cycle(turbine_id, issue, refresh_weather=live)
                    state["decisions"] = decisions
                    state["versions"] = [
                        ("План 23:00" if i == 0 else f"Обновление {scada_label(d['check_time'])[-5:]}",
                         read_version(agent, d["run_id"]))
                        for i, d in enumerate(decisions) if d.get("recomputed")
                    ]
                    state["result"] = state["versions"][0][1]
                    state["analysis"] = decisions[0]["analysis"]
                else:
                    result = agent.run(turbine_id, issue, refresh_weather=live)
                    state["result"], state["analysis"] = result, result.attrs["agent"]["analysis"]
            else:
                frames, days = [], []
                for target in pd.date_range(TEST_START, TEST_END, freq="1D"):
                    result = agent.run(turbine_id, issue_time_for(target.date()), refresh_weather=live)
                    frames.append(result.assign(forecast_date=target))
                    first_day = result.attrs["agent"]["analysis"]["days"][0]
                    days.append({"Сутки": f"{target:%d.%m}", "КИУМ, %": 100 * first_day["capacity_factor"],
                                 "Часы полной мощности": first_day["full_load_hours"], "Пик, %": 100 * first_day["peak_power"],
                                 "Ожидаемая ошибка, %": 100 * (first_day["expected_mae"] or np.nan)})
                state["february"], state["days"] = pd.concat(frames, ignore_index=True), pd.DataFrame(days)
        if mode != "Весь февраль" and llm != "off":
            with st.spinner("LLM-агент готовит брифинг (вызывает инструменты)…"):
                operator_agent = ForecastAgent(output_dir=session_output_dir() / "briefing", coordinates=cfg)
                checks = (6, 12) if mode == "Один выпуск" else ()
                state["briefing"] = OperatorAgent(operator_agent, llm).brief(turbine_id, state["issue"], checks=checks)
        st.session_state["run"] = state
    except Exception as exc:  # shown to the operator, logged in the agent log above
        st.session_state["run"] = None
        st.error(f"Прогноз не построен: {exc}")
    finally:
        st.session_state["log"] = lines
        for logger, level, propagate in saved:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate

state = st.session_state.get("run")
if st.session_state.get("log") and not run_clicked:
    with st.expander("Журнал агента: вызовы инструментов и решения", expanded=False):
        st.code("\n".join(st.session_state["log"]), language="text")

if state and state["key"] == request_key:
    if mode in ("Один выпуск", "Сейчас (live)"):
        issue = state["issue"]
        issued = issue.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)
        title = "Прогноз в реальном времени: 48 часов" if mode == "Сейчас (live)" else "Плановый выпуск: 48 часов"
        st.subheader(f"Турбина {turbine_id} · прогноз, выпущенный {issued:%d.%m.%Y %H:%M}")
        st.plotly_chart(forecast_figure(state["result"], state["analysis"], title), width="stretch")
        if state.get("briefing"):
            render_briefing(state["briefing"])
        render_analysis(state["analysis"])
        if state.get("decisions"):
            st.subheader("Повторный расчёт при обновлении входных данных")
            render_updates(state["decisions"])
            if len(state["versions"]) > 1:
                st.plotly_chart(versions_figure(state["versions"]), width="stretch")
    else:
        st.subheader(f"Турбина {turbine_id} · весь февраль (28 выпусков)")
        st.plotly_chart(heatmap_figure(state["february"]), width="stretch")
        st.markdown("**Прогноз на сутки D+1 по каждому выпуску**")
        st.dataframe(state["days"], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in state["days"].columns[1:]})
        export = state["february"].assign(time_scada=to_scada(state["february"]["time"]))
        st.download_button("Скачать прогнозы февраля (CSV)", export.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"turbine{turbine_id}_february_forecasts.csv", mime="text/csv")
elif not state:
    st.info("Выберите турбину и режим слева и нажмите «Запустить агента».")
