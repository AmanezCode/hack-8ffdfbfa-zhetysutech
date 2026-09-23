"""Fit the two-level model per turbine and report walk-forward validation."""

import argparse

import numpy as np
import pandas as pd

from src.config import HISTORY_START, HORIZON_HOURS, TEST_END, TURBINES, VALIDATION_DAYS
from src.data import load_turbine_hourly
from src.model import (
    build_training_table,
    fit_blend_weight,
    run_forecast_model,
    save_artifacts,
    train_residual_model,
)
from src.physics import PowerCurve, density_corrected_wind
from src.weather import load_weather

VALIDATION_START = pd.Timestamp("2026-01-31") - pd.Timedelta(days=VALIDATION_DAYS)


def daily_issue_times(history: pd.DataFrame) -> pd.DatetimeIndex:
    """One forecast per day, issued at 23:00 with that day's data in hand."""
    first = history.index[0].normalize() + pd.Timedelta(days=1, hours=23)
    last = history.index[-1].normalize() - pd.Timedelta(hours=1)
    return pd.date_range(first, last, freq="1D")


def evaluate(table: pd.DataFrame, booster, feature_columns, blend_weight: float) -> pd.DataFrame:
    residual = booster.predict(table[feature_columns])
    prediction = np.clip(table["level1"] + blend_weight * residual, 0.0, 1.0)
    actual = table["actual"]

    rows = {
        f"two-level (blend w={blend_weight:.2f})": prediction,
        "level 1 only (physics)": table["level1"],
        "persistence (last known hour)": table["power_lag_1"],
    }
    summary = pd.DataFrame(
        [
            {
                "model": name,
                "MAE": float(np.abs(values - actual).mean()),
                "RMSE": float(np.sqrt(((values - actual) ** 2).mean())),
            }
            for name, values in rows.items()
        ]
    )

    bucket = pd.cut(table["lead_hours"], [0, 12, 24, 36, 48], labels=["1-12h", "13-24h", "25-36h", "37-48h"])
    per_lead = pd.DataFrame(
        {
            "two-level": np.abs(prediction - actual).groupby(bucket, observed=True).mean(),
            "physics": np.abs(table["level1"] - actual).groupby(bucket, observed=True).mean(),
        }
    )
    return summary, per_lead


def run(turbine_id: int) -> None:
    from src.features import FEATURE_COLUMNS

    print(f"\n{'=' * 60}\nTurbine {turbine_id}\n{'=' * 60}")

    history = load_turbine_hourly(turbine_id)
    usable = history["power"].notna().sum()
    print(f"history: {history.index[0]} -> {history.index[-1]}  usable hours: {usable}/{len(history)}")

    weather = load_weather(turbine_id, HISTORY_START, TEST_END)
    print(f"weather: {weather.index[0]} -> {weather.index[-1]}  rows: {len(weather)}")

    issue_times = daily_issue_times(history)
    train_issues = issue_times[issue_times < VALIDATION_START]
    valid_issues = issue_times[issue_times >= VALIDATION_START]
    print(f"issue times: {len(train_issues)} train / {len(valid_issues)} validation")

    # The curve must be fitted on the same wind source it is fed at inference:
    # the Open-Meteo grid point, not the turbine anemometer. Fitting on observed
    # wind and predicting from forecast wind leaves Level 1 miscalibrated.
    fit_data = history.loc[history.index < VALIDATION_START].join(weather, how="inner")
    fit_data = fit_data.dropna(subset=["wind_fc", "temp_fc", "power"])
    corrected = density_corrected_wind(fit_data["wind_fc"], fit_data["temp_fc"])
    power_curve = PowerCurve().fit(corrected, fit_data["power"])
    print(f"power curve fitted on {len(fit_data)} hours of forecast wind")

    inner_split = train_issues[-60]
    fit_issues = train_issues[train_issues < inner_split]
    inner_issues = train_issues[train_issues >= inner_split]

    train_table = build_training_table(history, weather, power_curve, fit_issues, HORIZON_HOURS)
    inner_table = build_training_table(history, weather, power_curve, inner_issues, HORIZON_HOURS)
    valid_table = build_training_table(history, weather, power_curve, valid_issues, HORIZON_HOURS)
    print(f"training rows: {len(train_table)}  inner: {len(inner_table)}  holdout: {len(valid_table)}")

    booster = train_residual_model(train_table, inner_table)
    blend_weight = fit_blend_weight(booster, inner_table)
    print(f"best iteration: {booster.best_iteration}  blend weight: {blend_weight:.2f}")

    summary, per_lead = evaluate(valid_table, booster, FEATURE_COLUMNS, blend_weight)
    print(summary.to_string(index=False))
    print("\nMAE by lead time:")
    print(per_lead.to_string())

    save_artifacts(turbine_id, power_curve, booster, blend_weight)
    print(f"artifacts saved for turbine {turbine_id}")

    probe = run_forecast_model(
        history,
        weather,
        pd.Timestamp("2026-01-31 23:00"),
        {"power_curve": power_curve, "booster": booster, "blend_weight": blend_weight},
    )
    print(f"smoke test issue 2026-01-31 23:00 -> {len(probe)} hourly predictions, "
          f"mean {probe['prediction'].mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbine", type=int, choices=list(TURBINES), default=None)
    args = parser.parse_args()

    targets = [args.turbine] if args.turbine else list(TURBINES)
    for turbine_id in targets:
        run(turbine_id)


if __name__ == "__main__":
    main()
