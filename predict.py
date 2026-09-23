"""Historical replay of the February test: one 48 h forecast per day per turbine.

Issued at 23:00 on the SCADA clock of 31 January, then daily through
27 February; each run only sees weather published before its issue time.

All-or-nothing: the full set is built in memory first. If any issue cannot be
produced (no admissible weather even after falling back to older runs), the
script exits non-zero and the previous forecasts stay untouched. Retrying the
weather request or switching source is the agent's decision, not this runner's.
"""

import argparse
import shutil
import sys

import pandas as pd

from src.config import FORECASTS, ISSUE_HOUR_UTC, TEST_END, TEST_START, TURBINES
from src.data import load_turbine_hourly, utc_to_scada
from src.model import IncompleteWeatherError, load_artifacts, run_forecast_model
from src.weather import load_weather


def issue_times() -> pd.DatetimeIndex:
    first = pd.Timestamp(TEST_START) - pd.Timedelta(days=1)
    last = pd.Timestamp(TEST_END) - pd.Timedelta(days=1)
    return pd.date_range(first, last, freq="1D") + pd.Timedelta(hours=ISSUE_HOUR_UTC)


def build(turbine_id: int) -> tuple[dict[str, pd.DataFrame], list[str]]:
    history = load_turbine_hourly(turbine_id)
    weather = load_weather(turbine_id)
    artifacts = load_artifacts(turbine_id)
    model_version = artifacts["manifest"]["model_version"]

    outputs, failures = {}, []
    for issue_time in issue_times():
        issue_scada = utc_to_scada(issue_time)
        try:
            forecast = run_forecast_model(history, weather, issue_time, artifacts)
        except IncompleteWeatherError as error:
            hours = ", ".join(f"{t:%m-%d %H:%M}" for t in utc_to_scada(pd.DatetimeIndex(error.missing_hours))[:5])
            failures.append(f"turbine {turbine_id} issue {issue_scada:%Y-%m-%d %H:%M}: {error} ({hours}...)")
            continue

        forecast.index.name = "time_utc"
        outputs[f"turbine{turbine_id}_{issue_scada:%Y-%m-%d}.csv"] = forecast.assign(
            turbine_id=turbine_id,
            issue_time_utc=issue_time,
            issue_time_scada=issue_scada,
            model_version=model_version,
        )
        print(f"turbine {turbine_id} issue {issue_scada:%Y-%m-%d %H:%M} -> "
              f"{forecast['time_scada'].iloc[0]:%m-%d %H:%M}..{forecast['time_scada'].iloc[-1]:%m-%d %H:%M}, "
              f"mean {forecast['prediction'].mean():.3f}")

    if outputs:
        outputs[f"turbine{turbine_id}_february_all.csv"] = pd.concat(outputs.values())
    return outputs, failures


def publish(turbine_ids: list[int], outputs: dict[str, pd.DataFrame]) -> None:
    """Write the new set to a staging folder, then swap it in."""
    staging = FORECASTS.with_name(FORECASTS.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for name, frame in outputs.items():
        frame.to_csv(staging / name)

    FORECASTS.mkdir(parents=True, exist_ok=True)
    for turbine_id in turbine_ids:
        for old in FORECASTS.glob(f"turbine{turbine_id}_*.csv"):
            old.unlink()
    for path in staging.iterdir():
        path.replace(FORECASTS / path.name)
    staging.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbine", type=int, choices=list(TURBINES), default=None)
    args = parser.parse_args()
    turbine_ids = [args.turbine] if args.turbine else list(TURBINES)

    outputs, failures = {}, []
    for turbine_id in turbine_ids:
        turbine_outputs, turbine_failures = build(turbine_id)
        outputs.update(turbine_outputs)
        failures.extend(turbine_failures)

    if failures:
        print("\nreplay incomplete, previous forecasts left unchanged:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)

    publish(turbine_ids, outputs)
    expected = len(issue_times())
    print(f"\npublished {expected} issues x {len(turbine_ids)} turbines to {FORECASTS}")


if __name__ == "__main__":
    main()
