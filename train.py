"""Rolling-origin validation of four model variants, then a final fit per turbine.

Variants: physics (power curve on forecast wind), residual (curve + learned correction),
direct (LightGBM, three seeds averaged) and combined (direct averaged with the cascade
NWP -> wind at the anemometer -> anemometer power curve).

Folds are defined on target time (SCADA-clock months). An issue belongs to a
segment only if all 48 of its target hours fall inside it, so no label is shared
across the fit / inner / evaluation boundaries, and every label must have closed
before the first issue of the next segment. The power curve is fitted on the fit
segment only; the inner segment drives early stopping, the blend weight and the
interval width, and the scored models are then retrained on fit + inner with those
tree counts, as the final model is retrained on everything before the test period.

Training uses issues every 6 hours (the agent re-issues when fresher weather
arrives); scores are computed on the official daily issues at 23:00 SCADA.

Selection rule, fixed before looking at results: lowest mean MAE across folds.
"""

import argparse
import hashlib
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
    RUN_CYCLE_HOURS,
    SCADA_UTC_OFFSET_HOURS,
    TRAIN_ISSUE_HOURS_UTC,
    TURBINES,
    WEATHER_START,
)
from src.data import load_turbine_hourly
from src.features import FEATURE_COLUMNS, add_level1
from src.model import (
    CASCADE_SHARE,
    INTERVAL_COVERAGE,
    LGB_PARAMS,
    QUANTILES,
    SEEDS,
    VARIANTS,
    SeedEnsemble,
    build_training_table,
    fit_anemometer_curve,
    fit_blend_weight,
    fit_ensemble,
    fit_ensemble_rounds,
    fit_gbm,
    fit_interval_scale,
    fit_power_curve,
    interval,
    predict_variant,
    quantile_params,
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
        "within_10pct": float((np.abs(error) <= 0.10).mean()),
        "WAPE": float(np.abs(error).sum() / actual.sum()),
        "daily_energy_error": float((np.abs(daily["pred"] - daily["actual"]) / 24).mean()) if len(daily) else float("nan"),
    }


def update_gain(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    """MAE of the official forecast vs a re-issue k hours later, on the very same target hours."""
    rows = pd.DataFrame({"target": frame.index, "issue": frame["issue_time"].to_numpy(),
                         "error": np.abs(prediction - frame["actual"].to_numpy())})
    official_rows = rows[rows["issue"].dt.hour == ISSUE_HOUR_UTC]
    gains = {}
    for hours in (6, 12, 18):
        later = rows.assign(issue=rows["issue"] - pd.Timedelta(hours=hours))
        joined = official_rows.merge(later, on=["issue", "target"], suffixes=("_official", "_update"))
        gains[f"+{hours}h"] = {"pairs": int(len(joined)), "mae_official": float(joined["error_official"].mean()),
                               "mae_update": float(joined["error_update"].mean())}
    return gains


def refit(model, fit: pd.DataFrame, inner: pd.DataFrame, label_fit, label_inner, params: dict | None = None):
    """Retrain on fit + inner with the tree counts early stopping chose on the inner month."""
    both = pd.concat([fit, inner])
    label = np.concatenate([np.asarray(label_fit), np.asarray(label_inner)])
    if isinstance(model, SeedEnsemble):
        return fit_ensemble_rounds(both, label, model.best_iterations, params)
    return fit_gbm(both, label, num_round=max(int(model.best_iteration), 1), params=params)


def with_wind_label(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["wind_actual"].notna()]


def run_fold(table: pd.DataFrame, month: str, history: pd.DataFrame) -> dict:
    eval_start = scada_to_utc_ts(f"{month}-01")
    eval_end = scada_to_utc_ts((pd.Period(month, "M") + 1).strftime("%Y-%m-01"))
    inner_start = eval_start - pd.Timedelta(days=INNER_DAYS)

    # The same cutoff applies to evaluation: its scores choose the variant, and that
    # choice is made before the first issue of the following month.
    fit = labels_known_before(within(table, pd.Timestamp.min, inner_start), inner_start)
    inner = labels_known_before(within(table, inner_start, eval_start), eval_start)
    every_issue = labels_known_before(within(table, eval_start, eval_end), eval_end)
    evaluation = official(every_issue)

    curve = fit_power_curve(fit)
    for frame in (fit, inner, every_issue, evaluation):
        add_level1(frame, curve)

    # Early stopping on the inner month fixes the tree counts and the calibration below;
    # the scored models are then retrained on fit + inner, exactly as the final model is
    # retrained on every label before the test period.
    residual_label = (fit["actual"] - fit["level1"], inner["actual"] - inner["level1"])
    residual_booster = fit_gbm(fit, residual_label[0], inner, residual_label[1])
    blend = fit_blend_weight(residual_booster.predict(inner[FEATURE_COLUMNS]), inner)
    direct_model = fit_ensemble(fit, fit["actual"], inner, inner["actual"])
    # Cascade: the same features predict the hourly wind at the turbine anemometer.
    fit_wind, inner_wind = with_wind_label(fit), with_wind_label(inner)
    wind_model = fit_ensemble(fit_wind, fit_wind["wind_actual"], inner_wind, inner_wind["wind_actual"])
    boosters = {"residual": residual_booster, "direct": direct_model, "combined": direct_model}
    quantile_boosters = {name: fit_gbm(fit, fit["actual"], inner, inner["actual"], params=quantile_params(alpha))
                         for name, alpha in QUANTILES.items()}

    scored = {"residual": refit(residual_booster, fit, inner, *residual_label),
              "direct": refit(direct_model, fit, inner, fit["actual"], inner["actual"])}
    scored["combined"] = scored["direct"]
    scored_quantiles = {name: refit(quantile_boosters[name], fit, inner, fit["actual"], inner["actual"], quantile_params(alpha))
                        for name, alpha in QUANTILES.items()}
    # Each anemometer curve uses only labels closed before the data it is applied to.
    cascade_inner = (wind_model, fit_anemometer_curve(history, inner_start))
    cascade_eval = (refit(wind_model, fit_wind, inner_wind, fit_wind["wind_actual"], inner_wind["wind_actual"]),
                    fit_anemometer_curve(history, eval_start))

    predictions = {
        "physics": predict_variant("physics", evaluation),
        "residual": predict_variant("residual", evaluation, scored["residual"], blend),
        "direct": predict_variant("direct", evaluation, scored["direct"]),
        "combined": predict_variant("combined", evaluation, scored["combined"], cascade=cascade_eval),
        "climatology": evaluation["hour"].map(fit.groupby("hour")["actual"].mean()).to_numpy(),
    }

    # Interval: quantile models give the shape, the inner month calibrates the width, evaluation checks it.
    intervals = {}
    for name in ("residual", "direct", "combined"):
        q_inner = {q: b.predict(inner[FEATURE_COLUMNS]) for q, b in quantile_boosters.items()}
        scale = fit_interval_scale(predict_variant(name, inner, boosters[name], blend, cascade_inner),
                                   q_inner["q10"], q_inner["q90"], inner["actual"].to_numpy())
        q_eval = {q: b.predict(evaluation[FEATURE_COLUMNS]) for q, b in scored_quantiles.items()}
        low, high = interval(predictions[name], q_eval["q10"], q_eval["q90"], scale)
        actual_eval = evaluation["actual"].to_numpy()
        intervals[name] = {"scale": scale, "coverage": float(((actual_eval >= low) & (actual_eval <= high)).mean()),
                           "mean_width": float((high - low).mean())}
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
        "interval": intervals,
        "update_gain": {name: update_gain(every_issue, predict_variant(name, every_issue, scored.get(name), blend, cascade_eval))
                        for name in VARIANTS},
        "residual_iterations": int(residual_booster.best_iteration),
        "direct_iterations": direct_model.best_iterations,
        "wind_iterations": wind_model.best_iterations,
        # Hourly CV forecasts of every variant; saved for the chosen one, not kept in the manifest.
        "_cv_rows": pd.DataFrame({"issue_time": evaluation["issue_time"].to_numpy(), "lead_hours": lead, "actual": actual,
                                  **{name: predictions[name] for name in VARIANTS}}, index=evaluation.index.rename("time_utc")),
        "quantile_iterations": {name: int(b.best_iteration) for name, b in quantile_boosters.items()},
        "blend_weight": blend,
    }


def final_fit(table: pd.DataFrame, variant: str, folds: list[dict], history: pd.DataFrame) -> dict:
    """Refit on every label available before the test period with the settings chosen in CV."""
    test_start = scada_to_utc_ts("2026-02-01")
    train = labels_known_before(within(table, pd.Timestamp.min, test_start), test_start)
    curve = fit_power_curve(train)
    add_level1(train, curve)

    artifacts = {"variant": variant, "power_curve": curve, "blend_weight": 1.0, "booster": None,
                 "wind_model": None, "anemometer_curve": None, "quantiles": None, "interval_scale": 1.0}
    if variant == "residual":
        rounds = int(np.median([f["residual_iterations"] for f in folds]))
        artifacts["booster"] = fit_gbm(train, train["actual"] - train["level1"], num_round=max(rounds, 1))
        artifacts["blend_weight"] = float(np.mean([f["blend_weight"] for f in folds]))
    elif variant in ("direct", "combined"):
        # Tree count per seed: median over folds of what early stopping chose for that seed.
        rounds = np.median([f["direct_iterations"] for f in folds], axis=0)
        artifacts["booster"] = fit_ensemble_rounds(train, train["actual"], rounds)
    if variant == "combined":
        wind_rows = with_wind_label(train)
        artifacts["wind_model"] = fit_ensemble_rounds(wind_rows, wind_rows["wind_actual"],
                                                      np.median([f["wind_iterations"] for f in folds], axis=0))
        artifacts["anemometer_curve"] = fit_anemometer_curve(history, test_start)
    if variant != "physics":
        artifacts["quantiles"] = {
            name: fit_gbm(train, train["actual"], num_round=max(int(np.median([f["quantile_iterations"][name] for f in folds])), 1),
                          params=quantile_params(alpha))
            for name, alpha in QUANTILES.items()
        }
        artifacts["interval_scale"] = round(float(np.mean([f["interval"][variant]["scale"] for f in folds])), 2)
    artifacts["train_rows"] = len(train)
    artifacts["train_target_end"] = train.index.max()
    return artifacts


def content_hash(artifacts: dict, weather_sha: str) -> str:
    """Changes whenever weights, curve, features, parameters or the weather source change."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"variant": artifacts["variant"], "features": FEATURE_COLUMNS, "params": LGB_PARAMS,
                              "blend": artifacts["blend_weight"], "interval_scale": artifacts["interval_scale"],
                              "weather": weather_sha}, sort_keys=True).encode())
    for curve in (artifacts["power_curve"], artifacts.get("anemometer_curve")):
        if curve is not None:
            digest.update(curve.centers_.tobytes())
            digest.update(curve.values_.tobytes())
    for booster in [artifacts["booster"], artifacts.get("wind_model"), *(artifacts["quantiles"] or {}).values()]:
        if booster is not None:
            digest.update(booster.model_to_string().encode())
    return digest.hexdigest()[:10]


def mean_scores(folds: list[dict], name: str) -> dict:
    keys = folds[0]["scores"][name].keys()
    return {key: float(np.mean([f["scores"][name][key] for f in folds])) for key in keys}


def print_fold(fold: dict) -> None:
    print(f"\n  {fold['month']}: {fold['pairs']} pairs / {fold['unique_targets']} unique target hours, "
          f"blend w={fold['blend_weight']:.2f}, iters residual={fold['residual_iterations']} direct={fold['direct_iterations']} "
          f"wind={fold['wind_iterations']}")
    for name, s in fold["scores"].items():
        print(f"    {name:12s} MAE {s['MAE']:.4f}  RMSE {s['RMSE']:.4f}  accuracy {s['accuracy']:.1%}  "
              f"within 10% {s['within_10pct']:.1%}  daily energy err {s['daily_energy_error']:.4f}")
    for name, i in fold["interval"].items():
        print(f"    80% interval ({name}): coverage {i['coverage']:.1%}, width {i['mean_width']:.1%}, scale {i['scale']}")


def run(turbine_id: int) -> dict:
    print(f"\n{'=' * 70}\nTurbine {turbine_id}\n{'=' * 70}")
    started = time.perf_counter()
    history = load_turbine_hourly(turbine_id)
    weather = load_weather(turbine_id)

    table = build_training_table(history, weather, issue_times(WEATHER_START, HISTORY_END))
    print(f"labelled (issue, target) pairs: {len(table)}  issues: {table['issue_time'].nunique()}  "
          f"targets {table.index.min()} .. {table.index.max()} UTC")

    folds = [run_fold(table, month, history) for month in EVAL_MONTHS]
    for fold in folds:
        print_fold(fold)
    cv_rows = pd.concat([fold.pop("_cv_rows").assign(month=fold["month"]) for fold in folds])

    cv_mean = {name: mean_scores(folds, name) for name in (*VARIANTS, *BASELINES)}
    chosen = min(VARIANTS, key=lambda v: cv_mean[v]["MAE"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cv_rows[["month", "issue_time", "lead_hours", "actual", chosen]].rename(columns={chosen: "prediction"}).to_csv(
        ARTIFACTS / f"cv_predictions_t{turbine_id}.csv", float_format="%.6f", lineterminator="\n")
    print("\n  mean over folds: " + "  ".join(f"{v} MAE {cv_mean[v]['MAE']:.4f}" for v in (*VARIANTS, *BASELINES)))
    print(f"  selected by mean MAE: {chosen}  (accuracy {cv_mean[chosen]['accuracy']:.1%})")

    artifacts = final_fit(table, chosen, folds, history)
    train_seconds = time.perf_counter() - started
    provenance = weather_provenance(turbine_id)
    chosen_interval = [f["interval"][chosen] for f in folds] if chosen != "physics" else []
    gains = {k: {key: float(np.mean([f["update_gain"][chosen][k][key] for f in folds])) for key in ("mae_official", "mae_update")}
             for k in folds[0]["update_gain"][chosen]}
    for k, gain in gains.items():
        gain["improvement"] = 1 - gain["mae_update"] / gain["mae_official"]
    print("  re-issue gain on the same hours: " + ", ".join(f"{k} {g['improvement']:.1%}" for k, g in gains.items()))
    if chosen_interval:
        print(f"  80% interval: CV coverage {np.mean([i['coverage'] for i in chosen_interval]):.1%}, "
              f"mean width {np.mean([i['mean_width'] for i in chosen_interval]):.1%}")
    manifest = {
        "model_version": f"t{turbine_id}-{chosen}-{content_hash(artifacts, provenance['sha256'])}",
        "turbine_id": turbine_id,
        "variant": chosen,
        "selection_rule": "lowest mean MAE over rolling monthly folds " + ", ".join(EVAL_MONTHS),
        "cv_mean": cv_mean,
        "cv_folds": folds,
        "expected_abs_error_by_lead": np.mean([f["mae_by_lead_hour"][chosen] for f in folds], axis=0).round(4).tolist(),
        "deviation_q99": float(np.mean([f["deviation_q99"][chosen] for f in folds])),
        "interval": {
            "nominal_coverage": INTERVAL_COVERAGE,
            "scale": artifacts["interval_scale"],
            "cv_coverage": float(np.mean([i["coverage"] for i in chosen_interval])) if chosen_interval else None,
            "cv_mean_width": float(np.mean([i["mean_width"] for i in chosen_interval])) if chosen_interval else None,
        },
        "update_gain": gains,
        "blend_weight": artifacts["blend_weight"],
        "seeds": list(SEEDS),
        "cascade_share": CASCADE_SHARE if chosen == "combined" else None,
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
            "availability_rule": f"freshest previous_dayN whose run (latest {RUN_CYCLE_HOURS}-hourly start <= valid - N*24 h) "
                                 f"was public {PUBLICATION_DELAY_HOURS} h after start, by issue time",
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
