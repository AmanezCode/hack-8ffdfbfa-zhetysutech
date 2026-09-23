"""Streamlit dashboard for the repository's current ForecastAgent contract."""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    ARTIFACTS,
    FORECASTS,
    ISSUE_HOUR_UTC,
    SCADA_UTC_OFFSET_HOURS,
    TEST_END,
    TEST_START,
    TURBINES,
)
from src.forecast_agent import ForecastAgent

SCADA_TZ = timezone(timedelta(hours=SCADA_UTC_OFFSET_HOURS), name="SCADA UTC+6")
AGENT_LOGGER = logging.getLogger("src.forecast_agent")


class AgentLogHandler(logging.Handler):
    """Translate ForecastAgent's structured logs into the UI tool log."""

    def __init__(self, callback):
        super().__init__(level=logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("loading_weather"):
            line = f"[TOOL] get_archived_weather_forecast — {message}"
        elif message.startswith("running_model"):
            line = f"[TOOL] run_forecast_model — {message}"
        elif message.startswith("saved"):
            line = f"[TOOL] save_forecast — {message}"
        else:
            line = f"[AGENT] {record.levelname}: {message}"
        self.callback(line)


def issue_time_for(forecast_date: date | pd.Timestamp) -> pd.Timestamp:
    """Return 23:00 on the previous SCADA day as an aware UTC timestamp."""
    target = pd.Timestamp(forecast_date).normalize()
    issue_utc = target - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR_UTC)
    return issue_utc.tz_localize("UTC")


def add_display_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Время (UTC+6)"] = pd.to_datetime(result["time"], utc=True).dt.tz_convert(SCADA_TZ)
    return result


def run_february(agent: ForecastAgent, turbine_id: int, log) -> pd.DataFrame:
    outputs = []
    for target_date in pd.date_range(TEST_START, TEST_END, freq="1D"):
        issue_time = issue_time_for(target_date)
        log(f"[TOOL] ForecastAgent.run(turbine={turbine_id}, issue_time={issue_time.isoformat()})")
        result = agent.run(turbine_id, issue_time)
        result = result.assign(
            turbine_id=turbine_id,
            issue_time=issue_time,
            forecast_date=target_date,
        )
        outputs.append(result)
        log(
            f"[AGENT] {target_date:%Y-%m-%d}: {len(result)} часовых значений; "
            f"средняя мощность {result['prediction'].mean():.3f}; "
            f"сохранено {agent.last_saved_path}."
        )
    return pd.concat(outputs, ignore_index=True)


def render_backtest(turbine_id: int) -> None:
    st.subheader("Качество модели · январский backtest")
    manifest_path = ARTIFACTS / f"manifest_t{turbine_id}.json"
    if not manifest_path.is_file():
        st.info(f"Не найден manifest модели: {manifest_path.name}.")
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        january = next(fold for fold in manifest["cv_folds"] if fold["month"] == "2026-01")
        variant = manifest["variant"]
        selected_score = january["scores"][variant]
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        st.info(f"Не удалось прочитать январские метрики ({exc}).")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("MAE · январь", f"{100 * selected_score['MAE']:.2f}%")
    metric_cols[1].metric("RMSE · январь", f"{100 * selected_score['RMSE']:.2f}%")
    metric_cols[2].metric("Выбранная модель", variant)
    st.caption(
        f"Fold {january['month']}, {january['pairs']} issue/target pairs. "
        "Значения показаны в процентах нормализованной мощности."
    )

    scores = january.get("scores", {})
    comparison = pd.DataFrame(
        [
            {"Модель": name, "MAE, %": 100 * values["MAE"], "RMSE, %": 100 * values["RMSE"]}
            for name, values in scores.items()
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    with st.expander("Средние CV-метрики за ноябрь 2025 — январь 2026"):
        cv_mean = manifest.get("cv_mean", {})
        cv_table = pd.DataFrame(
            [
                {"Модель": name, "MAE, %": 100 * values["MAE"], "RMSE, %": 100 * values["RMSE"]}
                for name, values in cv_mean.items()
            ]
        )
        st.dataframe(cv_table, hide_index=True, use_container_width=True)
        st.caption(f"Версия: {manifest.get('model_version', 'unknown')}")


st.set_page_config(page_title="Wind Forecast Agent", page_icon="🌬️", layout="wide")
st.title("Wind Forecast Agent")
st.caption("Почасовой прогноз мощности ВЭС на 48 часов · SCADA UTC+6")

with st.sidebar:
    st.header("Параметры")
    turbine_id = st.selectbox(
        "Турбина",
        options=list(TURBINES),
        format_func=lambda item: f"Турбина {item}",
    )
    mode = st.radio("Период прогноза", ["Одна дата", "Весь февраль"])
    forecast_date = None
    if mode == "Одна дата":
        forecast_date = st.date_input(
            "Дата прогноза",
            value=date.fromisoformat(TEST_START),
            min_value=date.fromisoformat(TEST_START),
            max_value=date.fromisoformat(TEST_END),
        )
    run_clicked = st.button("RUN", type="primary", use_container_width=True)

request_key = (turbine_id, mode, forecast_date)
if run_clicked:
    log_lines: list[str] = []
    log_panel = st.empty()

    def show_log(line: str) -> None:
        log_lines.append(line)
        log_panel.code("\n".join(log_lines), language="text")

    handler = AgentLogHandler(show_log)
    old_level, old_propagate = AGENT_LOGGER.level, AGENT_LOGGER.propagate
    AGENT_LOGGER.setLevel(logging.INFO)
    AGENT_LOGGER.propagate = False
    AGENT_LOGGER.addHandler(handler)
    st.session_state["forecast_result"] = None
    try:
        coords = {tid: (cfg["lat"], cfg["lon"]) for tid, cfg in TURBINES.items()}
        agent = ForecastAgent(
            output_dir=FORECASTS / "streamlit",
            artifacts_dir=ARTIFACTS,
            coordinates=coords,
        )
        with st.spinner("ForecastAgent выполняет прогноз…"):
            if mode == "Одна дата":
                issue_time = issue_time_for(forecast_date)
                show_log(
                    f"[TOOL] ForecastAgent.run(turbine={turbine_id}, "
                    f"issue_time={issue_time.isoformat()})"
                )
                result = agent.run(turbine_id, issue_time)
                result = result.assign(
                    turbine_id=turbine_id,
                    issue_time=issue_time,
                    forecast_date=pd.Timestamp(forecast_date),
                )
            else:
                result = run_february(agent, turbine_id, show_log)
        st.session_state["forecast_result"] = {"key": request_key, "data": result}
        st.session_state["agent_logs"] = log_lines
    except Exception as exc:
        st.session_state["agent_logs"] = log_lines
        st.error(f"Не удалось построить прогноз: {exc}")
    finally:
        AGENT_LOGGER.removeHandler(handler)
        AGENT_LOGGER.setLevel(old_level)
        AGENT_LOGGER.propagate = old_propagate

saved_logs = st.session_state.get("agent_logs", [])
if saved_logs and not run_clicked:
    with st.expander("Лог ForecastAgent", expanded=True):
        st.code("\n".join(saved_logs), language="text")

render_backtest(turbine_id)

run_state = st.session_state.get("forecast_result")
if run_state and run_state["key"] == request_key:
    forecast = add_display_time(run_state["data"])
    st.subheader("Прогноз мощности")

    if mode == "Одна дата":
        chart_data = forecast.copy()
        chart_data["Мощность, %"] = chart_data["prediction"] * 100
        fig = px.line(
            chart_data,
            x="Время (UTC+6)",
            y="Мощность, %",
            title=f"Турбина {turbine_id} · {forecast_date:%d.%m.%Y}, 48 часов",
            markers=True,
        )
        fig.update_yaxes(rangemode="tozero", ticksuffix="%")
        fig.update_layout(xaxis_title="Время (SCADA, UTC+6)", yaxis_title="Нормализованная мощность")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            chart_data[["Время (UTC+6)", "prediction", "wind_fc", "lead_hours"]].rename(
                columns={
                    "prediction": "Прогноз мощности (0–1)",
                    "wind_fc": "Прогноз ветра (м/с)",
                    "lead_hours": "Горизонт (ч)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        pivot = forecast.pivot(index="forecast_date", columns="lead_hours", values="prediction").sort_index()
        heatmap = px.imshow(
            pivot * 100,
            aspect="auto",
            origin="lower",
            color_continuous_scale="Viridis",
            labels={
                "x": "Час прогноза (lead)",
                "y": "Дата прогноза (UTC+6)",
                "color": "Мощность, %",
            },
            title="48-часовые прогнозы для каждого дня февраля",
        )
        heatmap.update_layout(xaxis_title="Час прогноза (1–48)", yaxis_title="Дата прогноза")
        st.plotly_chart(heatmap, use_container_width=True)

        forecast_dates = sorted(pd.Timestamp(value) for value in forecast["forecast_date"].unique())
        selected_forecast_date = st.selectbox(
            "Показать отдельный 48-часовой прогноз",
            options=forecast_dates,
            format_func=lambda item: item.strftime("%d.%m.%Y"),
        )
        selected = forecast[forecast["forecast_date"] == selected_forecast_date].copy()
        selected["Мощность, %"] = selected["prediction"] * 100
        detail_fig = px.line(
            selected,
            x="Время (UTC+6)",
            y="Мощность, %",
            title=f"Турбина {turbine_id} · {selected_forecast_date:%d.%m.%Y}",
            markers=True,
        )
        detail_fig.update_yaxes(rangemode="tozero", ticksuffix="%")
        st.plotly_chart(detail_fig, use_container_width=True)

        csv_data = forecast.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Скачать прогнозы февраля (CSV)",
            data=csv_data,
            file_name=f"turbine{turbine_id}_february_forecasts.csv",
            mime="text/csv",
        )
