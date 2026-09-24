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
VARIANTS = ("physics", "residual", "direct", "combined")
# Point models are averages of boosters that differ only in their random seed.
SEEDS = (42, 43, 44)
# "combined": equal-weight average of the direct model and the cascade (NWP -> wind at the
# turbine anemometer -> anemometer power curve), two models that err differently.
CASCADE_SHARE = 0.5
# Normal operation for the anemometer power curve: not stopped with wind, not curtailed.
DOWNTIME_WIND, DOWNTIME_POWER = 4.5, 0.02
CURTAILED_WIND, CURTAILED_GAP = (5.0, 20.0), 0.3


class SeedEnsemble:
    """Mean of LightGBM boosters that differ only in their seed, used wherever a booster is."""

    SEPARATOR = "\n# ---- next seed ----\n"

    def __init__(self, boosters):
        self.boosters = list(boosters)

    def predict(self, x) -> np.ndarray:
        return np.mean([booster.predict(x) for booster in self.boosters], axis=0)

    def feature_name(self) -> list[str]:
        return self.boosters[0].feature_name()

    @property
    def best_iterations(self) -> list[int]:
        return [int(booster.best_iteration) for booster in self.boosters]

    def set_threads(self, threads: int) -> None:
        for booster in self.boosters:
            booster.params["num_threads"] = threads

    def model_to_string(self) -> str:
        return self.SEPARATOR.join(booster.model_to_string() for booster in self.boosters)

    def save_model(self, path) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.model_to_string())

    @classmethod
    def from_string(cls, text: str) -> "SeedEnsemble":
        return cls(lgb.Booster(model_str=part) for part in text.split(cls.SEPARATOR))


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
        # Second label for the cascade: hourly wind at the turbine anemometer (never a feature).
        features["wind_actual"] = history["wind_speed"].reindex(features.index).to_numpy()
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


def fit_ensemble(train: pd.DataFrame, label, valid: pd.DataFrame, valid_label, params: dict | None = None) -> SeedEnsemble:
    """One early-stopped booster per seed."""
    return SeedEnsemble(fit_gbm(train, label, valid, valid_label, params=(params or {}) | {"seed": seed}) for seed in SEEDS)


def fit_ensemble_rounds(train: pd.DataFrame, label, rounds, params: dict | None = None) -> SeedEnsemble:
    """One booster per seed with a fixed tree count each (from early stopping elsewhere)."""
    return SeedEnsemble(fit_gbm(train, label, num_round=max(int(r), 1), params=(params or {}) | {"seed": seed})
                        for seed, r in zip(SEEDS, rounds, strict=True))


def fit_anemometer_curve(history: pd.DataFrame, before: pd.Timestamp) -> PowerCurve:
    """Anemometer wind -> power from hours of normal operation whose label closed before `before`.

    Hours stopped despite wind (downtime) or far below a first-pass curve (curtailment)
    are left out: the curve describes what the turbine yields when it runs.
    """
    past = history.loc[: before - pd.Timedelta(hours=2)].dropna(subset=["wind_speed", "power"])
    wind, power = past["wind_speed"].to_numpy(), past["power"].to_numpy()
    first = PowerCurve().fit(wind, power)
    downtime = (wind >= DOWNTIME_WIND) & (power <= DOWNTIME_POWER)
    curtailed = (wind >= CURTAILED_WIND[0]) & (wind <= CURTAILED_WIND[1]) & (power < first.predict(wind) - CURTAILED_GAP)
    normal = ~(downtime | curtailed)
    return PowerCurve().fit(wind[normal], power[normal])


def cascade_power(frame: pd.DataFrame, wind_model, anemometer_curve: PowerCurve) -> np.ndarray:
    """Forecast wind at the turbine, then power from its own curve. The curve is monotone,
    so the curve of the median wind is the median power."""
    return anemometer_curve.predict(np.clip(wind_model.predict(frame[FEATURE_COLUMNS]), 0.0, None))


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


def predict_variant(variant: str, frame: pd.DataFrame, booster=None, blend_weight: float = 1.0, cascade=None) -> np.ndarray:
    """`cascade` is (wind model, anemometer power curve), needed by the "combined" variant only."""
    level1 = frame["level1"].to_numpy()
    if variant == "physics":
        return level1
    raw = booster.predict(frame[FEATURE_COLUMNS])
    if variant == "residual":
        return np.clip(level1 + blend_weight * raw, 0.0, 1.0)
    if variant == "direct":
        return np.clip(raw, 0.0, 1.0)
    if variant == "combined":
        cascaded = cascade_power(frame, *cascade)
        return np.clip((1 - CASCADE_SHARE) * np.clip(raw, 0.0, 1.0) + CASCADE_SHARE * cascaded, 0.0, 1.0)
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

    prediction = predict_variant(artifacts["variant"], features, artifacts.get("booster"), artifacts.get("blend_weight", 1.0),
                                 (artifacts.get("wind_model"), artifacts.get("anemometer_curve")))
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
            "wind_model": directory / f"booster_wind_t{turbine_id}.txt",
            **{name: directory / f"booster_{name}_t{turbine_id}.txt" for name in QUANTILES}}


def save_artifacts(turbine_id: int, artifacts: dict, manifest: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {key: artifacts.get(key) for key in ("variant", "power_curve", "blend_weight", "interval_scale", "anemometer_curve")},
        ARTIFACTS / f"model_t{turbine_id}.joblib",
    )
    models = {"booster": artifacts.get("booster"), "wind_model": artifacts.get("wind_model"), **(artifacts.get("quantiles") or {})}
    for name, path in _booster_paths(turbine_id).items():
        if models.get(name) is not None:
            models[name].save_model(str(path))
        elif path.exists():
            path.unlink()
    (ARTIFACTS / f"manifest_t{turbine_id}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _read_booster(path) -> SeedEnsemble | None:
    # Git autocrlf changes byte offsets stored in LightGBM text models on Windows.
    # Universal-newline decoding restores LF before LightGBM parses the model.
    return SeedEnsemble.from_string(path.read_text(encoding="utf-8")) if path.exists() else None


def load_artifacts(turbine_id: int, directory=None) -> dict:
    directory = directory or ARTIFACTS
    bundle = joblib.load(directory / f"model_t{turbine_id}.joblib")
    paths = _booster_paths(turbine_id, directory)
    bundle["booster"] = _read_booster(paths["booster"])
    bundle["wind_model"] = _read_booster(paths["wind_model"])
    quantiles = {name: _read_booster(paths[name]) for name in QUANTILES}
    bundle["quantiles"] = quantiles if all(quantiles.values()) else None
    bundle["manifest"] = json.loads((directory / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))
    return bundle
