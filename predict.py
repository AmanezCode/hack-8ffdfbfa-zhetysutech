"""Walk-forward over the February test period: one forecast per day per turbine.

This is the plain ML-side runner. The agent (src/agent.py) wraps the same
functions with tool calls, validation and logging.
"""

import argparse

import pandas as pd

from src.config import FORECASTS, HISTORY_START, TEST_END, TEST_START, TURBINES
from src.data import load_turbine_hourly
from src.model import load_artifacts, run_forecast_model
from src.weather import load_weather


def issue_times() -> pd.DatetimeIndex:
    """First forecast is issued on 31 January for 1-2 February, then daily."""
    first = pd.Timestamp(TEST_START) - pd.Timedelta(days=1)
    last = pd.Timestamp(TEST_END) - pd.Timedelta(days=1)
    return pd.date_range(first, last, freq="1D") + pd.Timedelta(hours=23)


def run(turbine_id: int) -> pd.DataFrame:
    history = load_turbine_hourly(turbine_id)
    weather = load_weather(turbine_id, HISTORY_START, TEST_END)
    artifacts = load_artifacts(turbine_id)

    FORECASTS.mkdir(parents=True, exist_ok=True)
    collected = []

    for issue_time in issue_times():
        forecast = run_forecast_model(history, weather, issue_time, artifacts)
        if forecast is None:
            print(f"turbine {turbine_id} {issue_time:%Y-%m-%d}: skipped, no usable inputs")
            continue

        forecast = forecast.assign(turbine_id=turbine_id, issue_time=issue_time)
        forecast.to_csv(FORECASTS / f"turbine{turbine_id}_{issue_time:%Y-%m-%d}.csv")
        collected.append(forecast)
        print(f"turbine {turbine_id} {issue_time:%Y-%m-%d}: {len(forecast)}h, mean {forecast['prediction'].mean():.3f}")

    combined = pd.concat(collected)
    combined.to_csv(FORECASTS / f"turbine{turbine_id}_february_all.csv")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbine", type=int, choices=list(TURBINES), default=None)
    args = parser.parse_args()

    for turbine_id in [args.turbine] if args.turbine else list(TURBINES):
        combined = run(turbine_id)
        print(f"turbine {turbine_id}: {len(combined)} rows across {combined['issue_time'].nunique()} issue dates\n")


if __name__ == "__main__":
    main()
