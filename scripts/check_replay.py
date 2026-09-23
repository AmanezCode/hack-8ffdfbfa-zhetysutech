"""Acceptance checks for the February replay: provenance of weather, full
horizon, no duplicates, independence from recent SCADA and from later weather.

Run after train.py and predict.py; exits non-zero on the first violation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FORECASTS, HORIZON_HOURS, PUBLICATION_DELAY_HOURS, TURBINES  # noqa: E402
from src.data import load_turbine_hourly  # noqa: E402
from src.model import IncompleteWeatherError, load_artifacts, run_forecast_model  # noqa: E402
from src.weather import load_weather  # noqa: E402


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        sys.exit(1)


def main() -> None:
    for turbine_id in TURBINES:
        print(f"--- turbine {turbine_id}")
        forecast = pd.read_csv(FORECASTS / f"turbine{turbine_id}_february_all.csv", parse_dates=["time_utc", "issue_time_utc"])

        key = ["turbine_id", "issue_time_utc", "time_utc"]
        check(not forecast.duplicated(key).any(), "no duplicate (turbine_id, issue_time, valid_time)")
        check(forecast[["prediction", "wind_fc", "temp_fc"]].notna().all().all(), "no NaN in prediction or weather")
        check(forecast["prediction"].between(0, 1).all(), "predictions within [0, 1]")
        counts = forecast.groupby("issue_time_utc").size()
        check((counts == HORIZON_HOURS).all() and len(counts) == 28, f"28 issues x {HORIZON_HOURS} h")

        lead = (forecast["time_utc"] - forecast["issue_time_utc"]) / pd.Timedelta(hours=1)
        check((forecast["run_day"] * 24 >= lead + PUBLICATION_DELAY_HOURS).all(),
              f"every weather value from a run published >= {PUBLICATION_DELAY_HOURS} h before issue")

        shared = forecast.groupby("time_utc").filter(lambda g: len(g) > 1)
        spread = shared.groupby("time_utc")["wind_fc"].agg(lambda s: s.max() - s.min())
        check((spread > 0).mean() > 0.9, f"overlapping issues see different weather runs ({(spread > 0).mean():.0%} of shared hours)")

        february = forecast["time_utc"].between(pd.Timestamp("2026-01-31 18:00"), pd.Timestamp("2026-02-28 17:00"))
        check(forecast.loc[february, "time_utc"].nunique() == 28 * 24, "all 672 February hours covered")

        flat = pd.read_csv(FORECASTS / f"submission_turbine{turbine_id}.csv", parse_dates=["time_scada", "issue_time_scada"])
        expected_hours = pd.date_range("2026-02-01", periods=28 * 24, freq="1h")
        day_before = flat["time_scada"].dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=23)
        check(len(flat) == 672 and pd.DatetimeIndex(flat["time_scada"]).equals(expected_hours)
              and (flat["issue_time_scada"] == day_before).all(),
              "submission: 672 hours, each from the 23:00 issue of the previous day")
        check(((flat["p10"] <= flat["prediction"] + 1e-12) & (flat["prediction"] <= flat["p90"] + 1e-12)).all(),
              "submission interval brackets the forecast")

        history = load_turbine_hourly(turbine_id)
        weather = load_weather(turbine_id)
        artifacts = load_artifacts(turbine_id)
        issue = pd.Timestamp("2026-02-10 17:00")
        reference = run_forecast_model(history, weather, issue, artifacts)

        blanked = history.copy()
        blanked.loc[issue - pd.Timedelta(days=30):, ["wind_speed", "power", "temperature"]] = np.nan
        check(np.allclose(run_forecast_model(blanked, weather, issue, artifacts)["prediction"], reference["prediction"]),
              "forecast does not depend on recent SCADA")

        tampered = weather.copy()
        late = tampered.index > issue + pd.Timedelta(hours=HORIZON_HOURS + 1)
        tampered.loc[late] = tampered.loc[late] * 3 + 5
        check(np.allclose(run_forecast_model(history, tampered, issue, artifacts)["prediction"], reference["prediction"]),
              "weather beyond the horizon does not change the forecast")

        first_issue = pd.Timestamp("2026-01-31 17:00")
        label_end = pd.Timestamp(artifacts["manifest"]["train_target_end"])
        check(label_end + pd.Timedelta(hours=1) <= first_issue,
              f"final model labels closed before the first issue (last label {label_end}, closes {label_end + pd.Timedelta(hours=1)})")

        selection_end = max(pd.Timestamp(fold["eval_last_target_utc"]) for fold in artifacts["manifest"]["cv_folds"])
        check(selection_end + pd.Timedelta(hours=1) <= first_issue,
              f"model selection (CV) labels closed before the first issue (last {selection_end})")

        target = issue + pd.Timedelta(hours=5)
        freshest_missing = weather.copy()
        freshest_missing.loc[target, ["wind_speed_100m_previous_day1", "temperature_2m_previous_day1"]] = np.nan
        fallback = run_forecast_model(history, freshest_missing, issue, artifacts)
        check(len(fallback) == HORIZON_HOURS and fallback.loc[target, "run_day"] == 2,
              "missing freshest run falls back to an older admissible run")

        all_missing = weather.copy()
        all_missing.loc[target, [c for c in weather.columns if "previous_day" in c]] = np.nan
        try:
            run_forecast_model(history, all_missing, issue, artifacts)
            raised = False
        except IncompleteWeatherError as error:
            raised = list(error.missing_hours) == [target]
        check(raised, "no admissible run -> IncompleteWeatherError naming the hour, not a shorter forecast")


if __name__ == "__main__":
    main()
