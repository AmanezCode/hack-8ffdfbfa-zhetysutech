import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ARTIFACTS, HORIZON_HOURS
from src.data import utc_to_scada
from src.features import FEATURE_COLUMNS, build_features
from src.physics import PowerCurve

LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "seed": 42,
}
VARIANTS = ("physics", "residual", "direct")


class IncompleteWeatherError(RuntimeError):
    def __init__(self, missing_hours):
        self.missing_hours = list(missing_hours)
        super().__init__(f"weather missing for {len(self.missing_hours)} target hours")


def build_training_table(history, weather, issue_times, horizon=HORIZON_HOURS) -> pd.DataFrame:
    """One row per (issue_time, target hour) with an observed label: the inference shape."""
    frames = []
    for issue_time in issue_times:
        features = build_features(history, weather, issue_time, None, horizon)
        features["actual"] = history["power"].reindex(features.index).to_numpy()
        features["issue_time"] = issue_time
        features = features.dropna(subset=["actual", "wind_fc", "temp_fc"])
        if len(features):
            frames.append(features)
    if not frames:
        raise ValueError("no usable training samples")
    return pd.concat(frames)


def fit_power_curve(table: pd.DataFrame) -> PowerCurve:
    """Fitted on the forecast wind the model will actually receive, not on the
    turbine anemometer: the two have different distributions."""
    return PowerCurve().fit(table["wind_corrected"], table["actual"])


def fit_gbm(train: pd.DataFrame, label: np.ndarray, valid=None, valid_label=None, num_round: int = 2000,
            params: dict | None = None) -> lgb.Booster:
    params = LGB_PARAMS | (params or {})
    dataset = lgb.Dataset(train[FEATURE_COLUMNS], label=label)
    if valid is None:
        return lgb.train(params, dataset, num_boost_round=num_round)
    valid_set = lgb.Dataset(valid[FEATURE_COLUMNS], label=valid_label, reference=dataset)
    return lgb.train(
        params,
        dataset,
        num_boost_round=num_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


QUANTILES = {"q10": 0.1, "q90": 0.9}
INTERVAL_COVERAGE = QUANTILES["q90"] - QUANTILES["q10"]


def quantile_params(alpha: float) -> dict:
    return {"objective": "quantile", "alpha": alpha, "metric": "quantile"}


def interval(prediction: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """80 % interval around the point forecast from quantile-model distances, widened by `scale`.

    The scale is chosen on held-out data so the interval really covers 80 %
    (split-conformal calibration); the raw quantile models alone under-cover.
    """
    low = prediction - scale * np.maximum(prediction - q_low, 0.0)
    high = prediction + scale * np.maximum(q_high - prediction, 0.0)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def fit_interval_scale(prediction: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, actual: np.ndarray) -> float:
    for scale in np.arange(0.5, 4.01, 0.05):
        low, high = interval(prediction, q_low, q_high, scale)
        if ((actual >= low) & (actual <= high)).mean() >= INTERVAL_COVERAGE:
            return round(float(scale), 2)
    return 4.0


def fit_blend_weight(residual_pred: np.ndarray, frame: pd.DataFrame) -> float:
    """Share of the learned correction to trust, chosen on held-out data; 0 falls back to the curve."""
    weights = np.linspace(0.0, 1.0, 21)
    level1, actual = frame["level1"].to_numpy(), frame["actual"].to_numpy()
    errors = [np.abs(np.clip(level1 + w * residual_pred, 0.0, 1.0) - actual).mean() for w in weights]
    return float(weights[int(np.argmin(errors))])


def predict_variant(variant: str, frame: pd.DataFrame, booster=None, blend_weight: float = 1.0) -> np.ndarray:
    level1 = frame["level1"].to_numpy()
    if variant == "physics":
        return level1
    raw = booster.predict(frame[FEATURE_COLUMNS])
    if variant == "residual":
        return np.clip(level1 + blend_weight * raw, 0.0, 1.0)
    if variant == "direct":
        return np.clip(raw, 0.0, 1.0)
    raise ValueError(f"unknown variant {variant}")


def run_forecast_model(history, weather, issue_time, artifacts, horizon=HORIZON_HOURS) -> pd.DataFrame:
    """Forecast for issue_time+1 .. issue_time+horizon, index in UTC.

    Raises IncompleteWeatherError listing the target hours without weather
    instead of returning a shorter forecast.
    """
    features = build_features(history, weather, issue_time, artifacts["power_curve"], horizon)
    missing = features.index[features[["wind_fc", "temp_fc"]].isna().any(axis=1)]
    if len(missing):
        raise IncompleteWeatherError(missing)

    prediction = predict_variant(artifacts["variant"], features, artifacts.get("booster"), artifacts.get("blend_weight", 1.0))
    result = pd.DataFrame(
        {
            "time_scada": utc_to_scada(features.index),
            "lead_hours": features["lead_hours"].to_numpy(),
            "prediction": prediction,
            "level1": features["level1"].to_numpy(),
            "residual_pred": prediction - features["level1"].to_numpy(),
            "wind_fc": features["wind_fc"].to_numpy(),
            "temp_fc": features["temp_fc"].to_numpy(),
            "run_day": features["run_day"].to_numpy(),
        },
        index=features.index,
    )
    quantile_boosters = artifacts.get("quantiles") or {}
    if set(quantile_boosters) == set(QUANTILES):
        low, high = interval(prediction, quantile_boosters["q10"].predict(features[FEATURE_COLUMNS]),
                             quantile_boosters["q90"].predict(features[FEATURE_COLUMNS]), artifacts.get("interval_scale", 1.0))
        result["p10"], result["p90"] = low, high
    return result


def _booster_paths(turbine_id: int, directory=ARTIFACTS) -> dict:
    return {"booster": directory / f"booster_t{turbine_id}.txt",
            **{name: directory / f"booster_{name}_t{turbine_id}.txt" for name in QUANTILES}}


def save_artifacts(turbine_id: int, artifacts: dict, manifest: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {key: artifacts[key] for key in ("variant", "power_curve", "blend_weight", "interval_scale")},
        ARTIFACTS / f"model_t{turbine_id}.joblib",
    )
    models = {"booster": artifacts.get("booster"), **(artifacts.get("quantiles") or {})}
    for name, path in _booster_paths(turbine_id).items():
        if models.get(name) is not None:
            models[name].save_model(str(path))
        elif path.exists():
            path.unlink()
    (ARTIFACTS / f"manifest_t{turbine_id}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _read_booster(path) -> lgb.Booster | None:
    # Git autocrlf changes byte offsets stored in LightGBM text models on Windows.
    # Universal-newline decoding restores LF before LightGBM parses the model.
    return lgb.Booster(model_str=path.read_text(encoding="utf-8")) if path.exists() else None


def load_artifacts(turbine_id: int, directory=None) -> dict:
    directory = directory or ARTIFACTS
    bundle = joblib.load(directory / f"model_t{turbine_id}.joblib")
    paths = _booster_paths(turbine_id, directory)
    bundle["booster"] = _read_booster(paths["booster"])
    quantiles = {name: _read_booster(paths[name]) for name in QUANTILES}
    bundle["quantiles"] = quantiles if all(quantiles.values()) else None
    bundle["manifest"] = json.loads((directory / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))
    return bundle
