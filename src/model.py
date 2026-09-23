import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, HORIZON_HOURS
from src.features import FEATURE_COLUMNS, build_features

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
}


def build_training_table(history, weather, power_curve, issue_times, horizon=HORIZON_HOURS):
    """One row per (issue_time, lead_hour): exactly the shape seen at inference."""
    frames = []
    for issue_time in issue_times:
        features = build_features(history, weather, issue_time, power_curve, horizon)
        if features is None or features.empty:
            continue
        actual = history["power"].reindex(features.index)
        features = features.assign(actual=actual.to_numpy(), issue_time=issue_time)
        frames.append(features.dropna(subset=["actual"]))

    if not frames:
        raise ValueError("no usable training samples")

    table = pd.concat(frames)
    table["residual"] = table["actual"] - table["level1"]
    return table


def train_residual_model(table: pd.DataFrame, inner_valid: pd.DataFrame, num_round: int = 2000) -> lgb.Booster:
    """Early stopping on an inner split so the reported holdout stays untouched."""
    dataset = lgb.Dataset(table[FEATURE_COLUMNS], label=table["residual"])
    valid = lgb.Dataset(inner_valid[FEATURE_COLUMNS], label=inner_valid["residual"], reference=dataset)
    return lgb.train(
        LGB_PARAMS,
        dataset,
        num_boost_round=num_round,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def fit_blend_weight(booster: lgb.Booster, inner_valid: pd.DataFrame) -> float:
    """How much of the learned correction to trust, chosen on held-out data.

    The residual is dominated by weather-forecast error, so the correction can
    be worth little; this lets the system fall back to the physical curve
    instead of betting on a model that does not generalise.
    """
    residual = booster.predict(inner_valid[FEATURE_COLUMNS])
    level1 = inner_valid["level1"].to_numpy()
    actual = inner_valid["actual"].to_numpy()

    weights = np.linspace(0.0, 1.0, 21)
    errors = [np.abs(np.clip(level1 + w * residual, 0.0, 1.0) - actual).mean() for w in weights]
    return float(weights[int(np.argmin(errors))])


def run_forecast_model(history, weather, issue_time, artifacts, horizon=HORIZON_HOURS):
    """Level 1 physics curve + Level 2 learned residual, clipped to [0, 1]."""
    power_curve = artifacts["power_curve"]
    booster = artifacts["booster"]
    blend = artifacts.get("blend_weight", 1.0)

    features = build_features(history, weather, issue_time, power_curve, horizon)
    if features is None or features.empty:
        return None

    residual = blend * booster.predict(features[FEATURE_COLUMNS])
    prediction = np.clip(features["level1"].to_numpy() + residual, 0.0, 1.0)

    return pd.DataFrame(
        {
            "level1": features["level1"].to_numpy(),
            "residual_pred": residual,
            "prediction": prediction,
            "wind_fc": features["wind_fc"].to_numpy(),
            "lead_hours": features["lead_hours"].to_numpy(),
        },
        index=features.index,
    )


def save_artifacts(turbine_id: int, power_curve, booster, blend_weight: float) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"power_curve": power_curve, "blend_weight": blend_weight},
        ARTIFACTS / f"model_t{turbine_id}.joblib",
    )
    booster.save_model(str(ARTIFACTS / f"residual_t{turbine_id}.txt"))


def load_artifacts(turbine_id: int) -> dict:
    bundle = joblib.load(ARTIFACTS / f"model_t{turbine_id}.joblib")
    return {
        "power_curve": bundle["power_curve"],
        "blend_weight": bundle["blend_weight"],
        "booster": lgb.Booster(model_file=str(ARTIFACTS / f"residual_t{turbine_id}.txt")),
    }
