"""Orchestration layer used by the Streamlit forecasting interface."""

from __future__ import annotations

from datetime import date
from time import perf_counter
from typing import Callable, TypeVar

import pandas as pd

from src.config import HORIZON_HOURS, HISTORY_START, TEST_END, TEST_START, TURBINES
from src.data import load_turbine_hourly
from src.model import load_artifacts, run_forecast_model
from src.weather import load_weather

T = TypeVar("T")


class ForecastAgent:
    """Load inputs, call model tools, and report a validated forecast."""

    def __init__(self, log_callback: Callable[[str], None] | None = None) -> None:
        self.logs: list[str] = []
        self.log_callback = log_callback

    def _log(self, message: str) -> None:
        self.logs.append(message)
        if self.log_callback:
            self.log_callback(message)

    def _tool(self, name: str, action: Callable[[], T]) -> T:
        self._log(f"[TOOL] {name} — запуск")
        started = perf_counter()
        try:
            result = action()
        except Exception as exc:
            self._log(f"[TOOL] {name} — ошибка: {exc}")
            raise
        self._log(f"[TOOL] {name} — готово ({perf_counter() - started:.1f} с)")
        return result

    @staticmethod
    def _issue_time(forecast_date: date | str | pd.Timestamp) -> pd.Timestamp:
        target_date = pd.Timestamp(forecast_date).normalize()
        first_test_date = pd.Timestamp(TEST_START).normalize()
        last_test_date = pd.Timestamp(TEST_END).normalize()
        if target_date < first_test_date or target_date > last_test_date:
            raise ValueError(f"Дата должна быть в тестовом периоде {TEST_START} — {TEST_END}.")
        return target_date - pd.Timedelta(days=1) + pd.Timedelta(hours=23)

    @staticmethod
    def _february_issue_times() -> pd.DatetimeIndex:
        first = pd.Timestamp(TEST_START) - pd.Timedelta(days=1) + pd.Timedelta(hours=23)
        last = pd.Timestamp(TEST_END) - pd.Timedelta(days=1) + pd.Timedelta(hours=23)
        return pd.date_range(first, last, freq="1D")

    def forecast_day(self, turbine_id: int, forecast_date: date | str | pd.Timestamp) -> pd.DataFrame:
        """Forecast the selected February date and the following 47 hours."""
        self.logs = []
        if turbine_id not in TURBINES:
            raise ValueError(f"Неизвестная турбина: {turbine_id}.")
        issue_time = self._issue_time(forecast_date)
        return self._run(turbine_id, pd.DatetimeIndex([issue_time]))

    def forecast_february(self, turbine_id: int) -> pd.DataFrame:
        """Replay one 48-hour forecast per day across the February test period."""
        self.logs = []
        if turbine_id not in TURBINES:
            raise ValueError(f"Неизвестная турбина: {turbine_id}.")
        return self._run(turbine_id, self._february_issue_times())

    def _run(self, turbine_id: int, issue_times: pd.DatetimeIndex) -> pd.DataFrame:
        self._log(
            f"[AGENT] Турбина {turbine_id}: готовлю {len(issue_times)} "
            f"прогноз(а) по {HORIZON_HOURS} часов."
        )
        history = self._tool(
            f"load_turbine_hourly(turbine_id={turbine_id})",
            lambda: load_turbine_hourly(turbine_id),
        )

        first_hour = issue_times[0]
        last_padded_hour = issue_times[-1] + pd.Timedelta(hours=HORIZON_HOURS + 1)
        weather_start = max(first_hour.normalize(), pd.Timestamp(HISTORY_START))
        weather_end = last_padded_hour.strftime("%Y-%m-%d")
        weather = self._tool(
            f"load_weather(turbine_id={turbine_id}, {weather_start:%Y-%m-%d}…{weather_end})",
            lambda: load_weather(
                turbine_id,
                weather_start.strftime("%Y-%m-%d"),
                weather_end,
            ),
        )
        artifacts = self._tool(
            f"load_artifacts(turbine_id={turbine_id})",
            lambda: load_artifacts(turbine_id),
        )

        forecasts: list[pd.DataFrame] = []
        for issue_time in issue_times:
            forecast = self._tool(
                f"run_forecast_model(issue_time={issue_time:%Y-%m-%d %H:%M})",
                lambda issue_time=issue_time: run_forecast_model(
                    history, weather, issue_time, artifacts
                ),
            )
            if forecast is None or len(forecast) != HORIZON_HOURS:
                received = 0 if forecast is None else len(forecast)
                raise RuntimeError(
                    f"Для запуска {issue_time:%Y-%m-%d %H:%M} модель вернула {received} из "
                    f"{HORIZON_HOURS} часов. Проверьте полноту архива погоды и входных данных."
                )
            forecast = forecast.assign(
                turbine_id=turbine_id,
                issue_time=issue_time,
                forecast_date=(issue_time.normalize() + pd.Timedelta(days=1)),
            )
            forecasts.append(forecast)
            self._log(
                f"[AGENT] {issue_time + pd.Timedelta(days=1):%Y-%m-%d}: "
                f"получено {len(forecast)} часовых значений, "
                f"средняя мощность {forecast['prediction'].mean():.3f}."
            )

        combined = pd.concat(forecasts).sort_values(["issue_time", "lead_hours"])
        self._log(f"[AGENT] Прогноз готов: {len(combined)} значений.")
        return combined
