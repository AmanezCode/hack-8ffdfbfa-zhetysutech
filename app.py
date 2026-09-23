"""Streamlit dashboard for single-day and February wind-power forecasts."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent import ForecastAgent
from src.config import ARTIFACTS, TEST_END, TEST_START, TURBINES

st.set_page_config(page_title="Wind Forecast Agent", page_icon="🌬️", layout="wide")
st.title("Wind Forecast Agent")
st.caption("Почасовой прогноз мощности ВЭС на 48 часов")

with st.sidebar:
    st.header("Параметры")
    turbine_id = st.selectbox(
        "Турбина",
        options=list(TURBINES),
        format_func=lambda item: f"Турбина {item}",
    )
    mode = st.radio("Период прогноза", ["Одна дата", "Весь февраль"], horizontal=False)
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

    st.session_state["forecast_result"] = None
    try:
        agent = ForecastAgent(log_callback=show_log)
        with st.spinner("ForecastAgent выполняет прогноз…"):
            if mode == "Одна дата":
                result = agent.forecast_day(turbine_id, forecast_date)
            else:
                result = agent.forecast_february(turbine_id)
        st.session_state["forecast_result"] = {
            "key": request_key,
            "data": result,
        }
        st.session_state["agent_logs"] = log_lines
    except Exception as exc:
        st.session_state["agent_logs"] = log_lines
        st.error(f"Не удалось построить прогноз: {exc}")

saved_logs = st.session_state.get("agent_logs", [])
if saved_logs and not run_clicked:
    with st.expander("Лог ForecastAgent", expanded=True):
        st.code("\n".join(saved_logs), language="text")

metrics_path = ARTIFACTS / f"backtest_t{turbine_id}.json"
st.subheader("Качество модели · backtest на январе")
if metrics_path.is_file():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_rows = metrics.get("models", [])
        primary = next(
            row for row in model_rows if str(row.get("model", "")).startswith("two-level")
        )
        metric_cols = st.columns(3)
        metric_cols[0].metric("MAE", f"{100 * float(primary['MAE']):.2f}%")
        metric_cols[1].metric("RMSE", f"{100 * float(primary['RMSE']):.2f}%")
        metric_cols[2].metric("Турбина", str(turbine_id))
        st.caption(
            "Январские точки запуска: "
            f"{pd.Timestamp(metrics['validation_issue_start']):%d.%m.%Y} — "
            f"{pd.Timestamp(metrics['validation_issue_end']):%d.%m.%Y}. "
            "Значения показаны в процентах нормализованной мощности."
        )
        comparison = pd.DataFrame(model_rows)
        if not comparison.empty:
            comparison["MAE, %"] = comparison.pop("MAE") * 100
            comparison["RMSE, %"] = comparison.pop("RMSE") * 100
            st.dataframe(
                comparison.rename(columns={"model": "Модель"}),
                hide_index=True,
                use_container_width=True,
            )
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        st.info(f"Метрики не удалось прочитать ({exc}). Перезапустите python train.py.")
else:
    st.info("Метрик пока нет. Сначала выполните `python train.py`, чтобы обучить модель и сохранить январский backtest.")

run_state = st.session_state.get("forecast_result")
if run_state and run_state["key"] == request_key:
    forecast = run_state["data"]
    st.subheader("Прогноз мощности")

    if mode == "Одна дата":
        chart_data = forecast.reset_index(names="Время")
        chart_data["Мощность, %"] = chart_data["prediction"] * 100
        fig = px.line(
            chart_data,
            x="Время",
            y="Мощность, %",
            title=f"Турбина {turbine_id} · прогноз с {forecast_date:%d.%m.%Y}, 48 часов",
            markers=True,
        )
        fig.update_yaxes(rangemode="tozero", ticksuffix="%")
        fig.update_layout(xaxis_title="Время", yaxis_title="Нормализованная мощность")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            chart_data[["Время", "prediction", "wind_fc", "lead_hours"]].rename(
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
        month_data = forecast.reset_index(names="time")
        pivot = month_data.pivot(
            index="forecast_date", columns="lead_hours", values="prediction"
        )
        heatmap = px.imshow(
            pivot * 100,
            aspect="auto",
            origin="lower",
            color_continuous_scale="Viridis",
            labels={
                "x": "Час прогноза (lead)",
                "y": "Дата прогноза",
                "color": "Мощность, %",
            },
            title="48-часовые прогнозы для каждого дня февраля",
        )
        heatmap.update_layout(xaxis_title="Час прогноза (1–48)", yaxis_title="Дата запуска")
        st.plotly_chart(heatmap, use_container_width=True)

        forecast_dates = sorted(month_data["forecast_date"].unique())
        selected_forecast_date = st.selectbox(
            "Показать отдельный 48-часовой прогноз",
            options=forecast_dates,
            format_func=lambda item: pd.Timestamp(item).strftime("%d.%m.%Y"),
        )
        selected = month_data[month_data["forecast_date"] == selected_forecast_date].copy()
        selected["Мощность, %"] = selected["prediction"] * 100
        detail_fig = px.line(
            selected,
            x="time",
            y="Мощность, %",
            title=f"Турбина {turbine_id} · {pd.Timestamp(selected_forecast_date):%d.%m.%Y}",
            markers=True,
        )
        detail_fig.update_yaxes(rangemode="tozero", ticksuffix="%")
        st.plotly_chart(detail_fig, use_container_width=True)

        csv_data = month_data.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Скачать прогнозы февраля (CSV)",
            data=csv_data,
            file_name=f"turbine{turbine_id}_february_forecasts.csv",
            mime="text/csv",
        )
