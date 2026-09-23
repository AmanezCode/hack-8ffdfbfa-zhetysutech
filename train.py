"""Rolling-origin validation of three model variants, then a final fit per turbine.

Folds are defined on target time (SCADA-clock months). An issue belongs to a
segment only if all 48 of its target hours fall inside it, so no label is shared
across the fit / inner / evaluation boundaries, and every label must have closed
before the first issue of the next segment. The power curve is fitted on the fit
segment only; the inner segment drives early stopping and the blend weight.

Training uses issues every 6 hours (the agent re-issues when fresher weather
arrives); scores are computed on the official daily issues at 23:00 SCADA.

Selection rule, fixed before looking at results: lowest mean MAE across folds.
"""

import argparse
import json
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS,
    HISTORY_END,
    HORIZON_HOURS,
    ISSUE_HOUR_UTC,
    PUBLICATION_DELAY_HOURS,
    SCADA_UTC_OFFSET_HOURS,
    TRAIN_ISSUE_HOURS_UTC,
    TURBINES,
    WEATHER_START,
)
from src.data import load_turbine_hourly
from src.features import FEATURE_COLUMNS, add_level1
from src.model import (
    LGB_PARAMS,
    VARIANTS,
    build_training_table,
    fit_blend_weight,
    fit_gbm,
    fit_power_curve,
    predict_variant,
    save_artifacts,
)
from src.weather import load_weather, weather_provenance

EVAL_MONTHS = ["2025-11", "2025-12", "2026-01"]
INNER_DAYS = 30
BASELINES = ("climatology", "persistence")


def scada_to_utc_ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text) - pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def issue_times(start: str, end_scada: str) -> pd.DatetimeIndex:
    days = pd.date_range(pd.Timestamp(start).normalize(), scada_to_utc_ts(end_scada).normalize() + pd.Timedelta(days=1), freq="1D")
    stamps = [day + pd.Timedelta(hours=hour) for day in days for hour in TRAIN_ISSUE_HOURS_UTC]
    return pd.DatetimeIndex(sorted(stamps))


def within(table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows of issues whose whole horizon lies in [start, end)."""
    first_target = table["issue_time"] + pd.Timedelta(hours=1)
    last_target = table["issue_time"] + pd.Timedelta(hours=HORIZON_HOURS)
    return table[(first_target >= start) & (last_target < end)].copy()


def labels_known_before(frame: pd.DataFrame, segment_end: pd.Timestamp) -> pd.DataFrame:
    """Drop labels still open when the next segment's first forecast is issued.

    That issue is at segment_end - 1h, and the hour stamped H only closes at
    H+1h (it averages H:00..H:50), so the last usable label is segment_end - 2h.
    """
    first_next_issue = segment_end - pd.Timedelta(hours=1)
    return frame[frame.index + pd.Timedelta(hours=1) <= first_next_issue]


def official(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["issue_time"].dt.hour == ISSUE_HOUR_UTC]


def score(prediction: np.ndarray, frame: pd.DataFrame) -> dict:
    """Errors in normalised power, i.e. as a share of rated capacity.

    accuracy = 1 - MAE is the usual capacity-normalised skill figure for wind
    forecasts; WAPE relates the error to the energy actually produced; the daily
    energy error compares 24 h sums (the unit a dispatcher plans in).
    """
    actual = frame["actual"].to_numpy()
    error = prediction - actual
    mae = float(np.abs(error).mean())

    blocks = pd.DataFrame({"issue": frame["issue_time"].to_numpy(), "day": (frame["lead_hours"].to_numpy() - 1) // 24,
                           "pred": prediction, "actual": actual})
    daily = blocks.groupby(["issue", "day"]).agg(pred=("pred", "sum"), actual=("actual", "sum"), hours=("pred", "size"))
    daily = daily[daily["hours"] == 24]

    return {
        "MAE": mae,
        "RMSE": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "accuracy": 1.0 - mae,
        "WAPE": float(np.abs(error).sum() / actual.sum()),
        "daily_energy_error": float((np.abs(daily["pred"] - daily["actual"]) / 24).mean()) if len(daily) else float("nan"),
    }


def run_fold(table: pd.DataFrame, month: str) -> dict:
    eval_start = scada_to_utc_ts(f"{month}-01")
    eval_end = scada_to_utc_ts((pd.Period(month, "M") + 1).strftime("%Y-%m-01"))
    inner_start = eval_start - pd.Timedelta(days=INNER_DAYS)

    # The same cutoff applies to evaluation: its scores choose the variant, and that
    # choice is made before the first issue of the following month.
    fit = labels_known_before(within(table, pd.Timestamp.min, inner_start), inner_start)
    inner = labels_known_before(within(table, inner_start, eval_start), eval_start)
    evaluation = official(labels_known_before(within(table, eval_start, eval_end), eval_end))

    curve = fit_power_curve(fit)
    for frame in (fit, inner, evaluation):
        add_level1(frame, curve)

    residual_booster = fit_gbm(fit, fit["actual"] - fit["level1"], inner, inner["actual"] - inner["level1"])
    blend = fit_blend_weight(residual_booster.predict(inner[FEATURE_COLUMNS]), inner)
    direct_booster = fit_gbm(fit, fit["actual"], inner, inner["actual"])

    predictions = {
        "physics": predict_variant("physics", evaluation),
        "residual": predict_variant("residual", evaluation, residual_booster, blend),
        "direct": predict_variant("direct", evaluation, direct_booster),
        "climatology": evaluation["hour"].map(fit.groupby("hour")["actual"].mean()).to_numpy(),
    }
    actual = evaluation["actual"].to_numpy()
    lead = evaluation["lead_hours"].to_numpy()

    scores = {name: score(p, evaluation) for name, p in predictions.items()}
    has_lag = evaluation["power_lag_1"].notna().to_numpy()
    scores["persistence"] = score(evaluation["power_lag_1"].to_numpy()[has_lag], evaluation[has_lag])

    return {
        "month": month,
        "pairs": int(len(evaluation)),
        "unique_targets": int(evaluation.index.nunique()),
        "eval_last_target_utc": str(evaluation.index.max()),
        "fit_rows": int(len(fit)),
        "inner_rows": int(len(inner)),
        "scores": scores,
        "mae_by_lead": {
            label: {name: float(np.abs(p - actual)[(lead >= low) & (lead <= high)].mean()) for name, p in predictions.items()}
            for label, low, high in [("1-24h", 1, 24), ("25-48h", 25, 48)]
        },
        "mae_by_lead_hour": {
            name: [float(np.abs(predictions[name] - actual)[lead == h].mean()) for h in range(1, HORIZON_HOURS + 1)]
            for name in VARIANTS
        },
        "deviation_q99": {name: float(np.quantile(np.abs(predictions[name] - evaluation["level1"].to_numpy()), 0.99)) for name in VARIANTS},
        "residual_iterations": int(residual_booster.best_iteration),
        "direct_iterations": int(direct_booster.best_iteration),
        "blend_weight": blend,
    }


def final_fit(table: pd.DataFrame, variant: str, folds: list[dict]) -> dict:
    """Refit on every label available before the test period with the settings chosen in CV."""
    test_start = scada_to_utc_ts("2026-02-01")
    train = labels_known_before(within(table, pd.Timestamp.min, test_start), test_start)
    curve = fit_power_curve(train)
    add_level1(train, curve)

    artifacts = {"variant": variant, "power_curve": curve, "blend_weight": 1.0, "booster": None}
    if variant == "residual":
        rounds = int(np.median([f["residual_iterations"] for f in folds]))
        artifacts["booster"] = fit_gbm(train, train["actual"] - train["level1"], num_round=max(rounds, 1))
        artifacts["blend_weight"] = float(np.mean([f["blend_weight"] for f in folds]))
    elif variant == "direct":
        rounds = int(np.median([f["direct_iterations"] for f in folds]))
        artifacts["booster"] = fit_gbm(train, train["actual"], num_round=max(rounds, 1))
    artifacts["train_rows"] = len(train)
    artifacts["train_target_end"] = train.index.max()
    return artifacts


def mean_scores(folds: list[dict], name: str) -> dict:
    keys = folds[0]["scores"][name].keys()
    return {key: float(np.mean([f["scores"][name][key] for f in folds])) for key in keys}


def print_fold(fold: dict) -> None:
    print(f"\n  {fold['month']}: {fold['pairs']} pairs / {fold['unique_targets']} unique target hours, "
          f"blend w={fold['blend_weight']:.2f}, iters residual={fold['residual_iterations']} direct={fold['direct_iterations']}")
    for name, s in fold["scores"].items():
        print(f"    {name:12s} MAE {s['MAE']:.4f}  RMSE {s['RMSE']:.4f}  accuracy {s['accuracy']:.1%}  daily energy err {s['daily_energy_error']:.4f}")


def run(turbine_id: int) -> dict:
    print(f"\n{'=' * 70}\nTurbine {turbine_id}\n{'=' * 70}")
    started = time.perf_counter()
    history = load_turbine_hourly(turbine_id)
    weather = load_weather(turbine_id)

    table = build_training_table(history, weather, issue_times(WEATHER_START, HISTORY_END))
    print(f"labelled (issue, target) pairs: {len(table)}  issues: {table['issue_time'].nunique()}  "
          f"targets {table.index.min()} .. {table.index.max()} UTC")

    folds = [run_fold(table, month) for month in EVAL_MONTHS]
    for fold in folds:
        print_fold(fold)

    cv_mean = {name: mean_scores(folds, name) for name in (*VARIANTS, *BASELINES)}
    chosen = min(VARIANTS, key=lambda v: cv_mean[v]["MAE"])
    print("\n  mean over folds: " + "  ".join(f"{v} MAE {cv_mean[v]['MAE']:.4f}" for v in (*VARIANTS, *BASELINES)))
    print(f"  selected by mean MAE: {chosen}  (accuracy {cv_mean[chosen]['accuracy']:.1%})")

    artifacts = final_fit(table, chosen, folds)
    train_seconds = time.perf_counter() - started
    provenance = weather_provenance(turbine_id)
    manifest = {
        "model_version": f"t{turbine_id}-{chosen}-{provenance['sha256'][:8]}",
        "turbine_id": turbine_id,
        "variant": chosen,
        "selection_rule": "lowest mean MAE over rolling monthly folds " + ", ".join(EVAL_MONTHS),
        "cv_mean": cv_mean,
        "cv_folds": folds,
        "expected_abs_error_by_lead": np.mean([f["mae_by_lead_hour"][chosen] for f in folds], axis=0).round(4).tolist(),
        "deviation_q99": float(np.mean([f["deviation_q99"][chosen] for f in folds])),
        "blend_weight": artifacts["blend_weight"],
        "train_rows": artifacts["train_rows"],
        "train_seconds": round(train_seconds, 1),
        "train_target_start": str(table.index.min()),
        "train_target_end": str(artifacts["train_target_end"]),
        "data_cutoff_scada": HISTORY_END,
        "issue_hours_utc": {"official": ISSUE_HOUR_UTC, "training": list(TRAIN_ISSUE_HOURS_UTC)},
        "time_convention": f"internal UTC; SCADA clock = UTC+{SCADA_UTC_OFFSET_HOURS}; official issue at {ISSUE_HOUR_UTC}:00 UTC",
        "weather": {
            "endpoint": provenance["endpoint"],
            "grid": [provenance["grid_latitude"], provenance["grid_longitude"]],
            "retrieved_at": provenance["retrieved_at"],
            "sha256": provenance["sha256"],
            "availability_rule": f"previous_dayN with N = ceil((lead + {PUBLICATION_DELAY_HOURS}) / 24)",
        },
        "feature_schema": FEATURE_COLUMNS,
        "lgb_params": LGB_PARAMS,
        "versions": {"python": platform.python_version(), **{pkg: version(pkg) for pkg in ("pandas", "numpy", "lightgbm")}},
    }
    save_artifacts(turbine_id, artifacts, manifest)
    print(f"  final model: {manifest['model_version']} on {artifacts['train_rows']} rows in {train_seconds:.0f}s -> {ARTIFACTS}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbine", type=int, choices=list(TURBINES), default=None)
    args = parser.parse_args()

    summary = {}
    for turbine_id in [args.turbine] if args.turbine else list(TURBINES):
        manifest = run(turbine_id)
        summary[turbine_id] = {"variant": manifest["variant"], "cv_mean": manifest["cv_mean"]}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
