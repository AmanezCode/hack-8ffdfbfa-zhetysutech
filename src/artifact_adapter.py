"""Optional generic artifact adapter; the team's ML lives in src.model.

Turbine N uses model_tN.joblib, residual_tN.txt and metadata_tN.json::

    {"feature_names": ["wind_fc", "hour"],
     "residual": {"format": "scalar"}}

feature_names may be omitted when the estimator has feature_names_in_. If both
exist, their order must agree. For a LightGBM text residual use::

    {"residual": {"format": "lightgbm", "feature_names": ["wind_fc"]}}

Residual feature names are required and must match the booster's stored order.
Only deserialize trusted joblib files: loading them can execute arbitrary code.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _names(value: Any, label: str) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (not isinstance(value, list) or not value
            or any(not isinstance(name, str) or not name.strip() for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must be a nonempty list of unique feature names")
    return value


def _model_names(model: Any, explicit: Any) -> list[str]:
    stored = getattr(model, "feature_names_in_", None)
    if stored is None and explicit is None:
        raise ValueError("Estimator requires feature_names_in_ or metadata feature_names; order cannot be guessed")
    names = _names(stored if stored is not None else explicit, "model feature_names")
    if explicit is not None and names != _names(explicit, "metadata feature_names"):
        raise ValueError("metadata feature_names disagree with estimator feature_names_in_ order")
    if getattr(model, "n_features_in_", len(names)) != len(names):
        raise ValueError("Estimator feature count disagrees with feature_names")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Joblib estimator must implement predict")
    return names


def load_artifacts(artifacts_dir: str | Path = ROOT / "artifacts") -> dict[int, dict[str, Any]]:
    """Load complete pairs; skip missing pairs, reject malformed present artifacts.

    Missing directories return {}. Entries contain model, feature_names,
    residual_format, residual and (for LightGBM) residual_feature_names.
    LightGBM is imported only for an explicitly declared LightGBM residual.
    """
    directory = Path(artifacts_dir)
    result: dict[int, dict[str, Any]] = {}
    for model_path in sorted(directory.glob("model_t*.joblib")):
        match = re.fullmatch(r"model_t([1-9][0-9]*)\.joblib", model_path.name)
        if match is None:
            continue
        turbine_id = int(match[1])
        residual_path = directory / f"residual_t{turbine_id}.txt"
        if not residual_path.is_file():
            continue
        metadata_path = directory / f"metadata_t{turbine_id}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing {metadata_path}; residual format must be declared explicitly")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("residual"), dict):
                raise ValueError("metadata must be an object containing a residual object")
            config = metadata["residual"]
            residual_format = config.get("format")
            if residual_format not in ("scalar", "lightgbm"):
                raise ValueError("residual.format must be 'scalar' or 'lightgbm'")
            try:
                model = joblib.load(model_path)
            except Exception as exc:
                raise ValueError(f"Cannot load {model_path.name}: {exc}") from exc
            entry = {"model": model, "feature_names": _model_names(model, metadata.get("feature_names")),
                     "residual_format": residual_format}
            if residual_format == "scalar":
                residual = float(residual_path.read_text(encoding="utf-8").strip())
                if not np.isfinite(residual):
                    raise ValueError("scalar residual must be finite")
                entry["residual"] = residual
            else:
                names = _names(config.get("feature_names"), "residual.feature_names")
                try:
                    import lightgbm
                except ImportError as exc:
                    raise ImportError("Declared LightGBM residual requires optional package 'lightgbm'") from exc
                booster = lightgbm.Booster(model_file=str(residual_path))
                if names != booster.feature_name():
                    raise ValueError("residual.feature_names disagree with LightGBM feature order")
                entry.update(residual=booster, residual_feature_names=names)
            result[turbine_id] = entry
        except ImportError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid artifacts for turbine {turbine_id}: {exc}") from exc
    return result


def _matrix(weather: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    missing = [name for name in names if name not in weather]
    if missing:
        raise ValueError(f"Weather is missing model features: {missing}")
    try:
        matrix = weather.loc[:, names].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model features must be numeric and finite") from exc
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Model features must be numeric and finite")
    return matrix


def _output(value: Any, size: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric vector") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must have shape ({size},) and contain only finite values")
    return array


def run_forecast_model(
    turbine_id: int, issue_time: str | pd.Timestamp, weather: pd.DataFrame,
    artifacts: dict[int, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (level1, residual_pred, clip(level1 + residual_pred, 0, 1)).

    Weather rows retain order. Named numeric features are used directly.
    Reserved hour/day_of_year/lead_hours are derived from weather.time in UTC
    when requested; naive timestamps mean UTC. issue_time must always be valid.
    Empty weather is rejected. The caller's DataFrame is never modified.
    """
    artifacts = load_artifacts() if artifacts is None else artifacts
    if turbine_id not in artifacts:
        raise FileNotFoundError(f"Missing artifact pair for turbine {turbine_id}: model_t{turbine_id}.joblib and residual_t{turbine_id}.txt")
    if not isinstance(weather, pd.DataFrame) or weather.empty or not weather.columns.is_unique:
        raise ValueError("weather must be a nonempty DataFrame with unique columns")
    try:
        issue = pd.Timestamp(issue_time)
        if pd.isna(issue):
            raise ValueError("missing issue_time")
        issue = issue.tz_localize("UTC") if issue.tzinfo is None else issue.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError("issue_time must be a valid timestamp") from exc
    artifact = artifacts[turbine_id]
    model = artifact["model"]
    names = _model_names(model, artifact.get("feature_names"))
    residual_format = artifact.get("residual_format")
    if residual_format not in ("scalar", "lightgbm"):
        raise ValueError("Artifact residual_format must be explicitly scalar or lightgbm")
    residual_names = (_names(artifact.get("residual_feature_names"), "residual_feature_names")
                      if residual_format == "lightgbm" else [])
    features = weather.copy()
    time_names = set(names + residual_names) & {"hour", "day_of_year", "lead_hours"}
    if time_names:
        if "time" not in features:
            raise ValueError("Weather requires time for UTC time features")
        try:
            times = pd.to_datetime(features["time"], utc=True, errors="raise", format="mixed")
        except (TypeError, ValueError) as exc:
            raise ValueError("Weather time must contain valid UTC-convertible timestamps") from exc
        if times.isna().any():
            raise ValueError("Weather time contains missing timestamps")
        features["hour"] = times.dt.hour
        features["day_of_year"] = times.dt.dayofyear
        features["lead_hours"] = (times - issue).dt.total_seconds() / 3600
    matrix = _matrix(features, names)
    residual_matrix = _matrix(features, residual_names) if residual_names else None
    level1 = _output(model.predict(matrix), len(weather), "level1")
    if residual_format == "scalar":
        scalar = _output([artifact["residual"]], 1, "scalar residual")[0]
        residual_pred = np.full(len(weather), scalar)
    else:
        residual_pred = _output(artifact["residual"].predict(residual_matrix), len(weather), "residual_pred")
    with np.errstate(over="ignore", invalid="ignore"):
        combined = _output(level1 + residual_pred, len(weather), "combined prediction")
    return level1, residual_pred, np.clip(combined, 0.0, 1.0)
