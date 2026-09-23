from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout

from .artifact_adapter import load_artifacts, run_forecast_model
from .archived_weather import load_weather


ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["time", "level1", "residual_pred", "prediction", "wind_fc", "lead_hours"]
LOG = logging.getLogger(__name__)


def issue_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp != stamp.floor("h"):
        raise ValueError("issue_time must be a timezone-aware whole hour")
    return stamp.tz_convert("UTC")


class ForecastAgent:
    """Orchestrates archived weather, ML inference, validation and persistence."""

    def __init__(self, output_dir: str | Path = ROOT / "forecasts",
                 artifacts_dir: str | Path = ROOT / "artifacts", *,
                 coordinates: Mapping[int, tuple[float, float]] | None = None,
                 archive_path: str | Path | None = None,
                 model_backend: str = "team",
                 weather_loader: Callable | None = None,
                 model_runner: Callable | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)
        if model_backend not in ("team", "generic"):
            raise ValueError("model_backend must be team or generic")
        self.model_backend = model_backend
        self.coordinates = dict(coordinates or {})
        self.archive_path = archive_path
        self.weather_loader = weather_loader
        self.model_runner = model_runner
        self.last_saved_path: Path | None = None
        self._model_metadata: dict = {}

    def get_archived_weather_forecast(self, lat: float, lon: float, issue_date: str | pd.Timestamp) -> pd.DataFrame:
        stamp = issue_timestamp(issue_date)
        loader = self.weather_loader or load_weather
        if self.weather_loader is None and self.archive_path is None and self.model_backend == "team":
            from .ml_bridge import load_team_weather
            loader = load_team_weather
        weather = loader(lat, lon, stamp) if self.archive_path is None else loader(lat, lon, stamp, archive_path=self.archive_path)
        self.validate_weather(weather)
        expected = pd.date_range(stamp + pd.Timedelta(hours=1), periods=48, freq="h")
        if not pd.DatetimeIndex(weather["time"]).equals(expected):
            raise ValueError("Weather must cover issue_time + 1..48 hours in UTC")
        return weather

    @staticmethod
    def validate_weather(weather: pd.DataFrame) -> None:
        required = {"time", "wind_fc", "temperature"}
        missing = required - set(weather.columns)
        if missing:
            raise ValueError(f"Weather is missing columns: {sorted(missing)}")
        if len(weather) != 48:
            raise ValueError(f"Expected 48 hourly weather rows, got {len(weather)}")
        if weather[sorted(required)].isna().any().any():
            raise ValueError("Weather contains NaN values")
        if not weather["time"].is_monotonic_increasing or weather["time"].duplicated().any():
            raise ValueError("Weather timestamps must be unique and increasing")
        times = pd.DatetimeIndex(weather["time"])
        if times.tz is None or not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
            raise ValueError("Weather requires timezone-aware consecutive hourly timestamps")
        numeric = weather.select_dtypes(include="number")
        if weather.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Weather contains missing or nonfinite values")
        if not np.isfinite(weather[["wind_fc", "temperature"]].to_numpy(dtype=float)).all() or (weather["wind_fc"] < 0).any():
            raise ValueError("Weather wind/temperature must be finite and wind nonnegative")

    def _weather(self, turbine_id: int, issue_time: pd.Timestamp) -> pd.DataFrame:
        if turbine_id not in (1, 2) or turbine_id not in self.coordinates:
            raise ValueError("Provide verified coordinates for turbine 1 or 2")
        return self.get_archived_weather_forecast(*self.coordinates[turbine_id], issue_time)

    def _predict(self, turbine_id: int, issue_time: pd.Timestamp, weather: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_runner is not None:
            outputs = self.model_runner(turbine_id, issue_time, weather.copy(deep=True))
        elif self.model_backend == "generic":
            outputs = run_forecast_model(turbine_id, issue_time, weather.copy(deep=True), load_artifacts(self.artifacts_dir))
        else:
            from .ml_bridge import run_team_model
            team_result = run_team_model(turbine_id, issue_time, weather.copy(deep=True), self.artifacts_dir)
            self._model_metadata = team_result.attrs["model"]
            outputs = tuple(team_result[name].to_numpy() for name in ("level1", "residual_pred", "prediction"))
        if len(outputs) != 3:
            raise ValueError("Model must return level1, residual_pred, prediction")
        arrays = tuple(np.asarray(value, dtype=float) for value in outputs)
        if any(value.shape != (48,) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Each model output must be a finite vector of 48 values")
        return arrays

    def run_forecast_model(self, turbine_id: int, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stamp = issue_timestamp(issue_time)
        weather = self._weather(turbine_id, stamp)
        return self._predict(turbine_id, stamp, weather)

    @staticmethod
    def validate_prediction(pred: pd.DataFrame, weather: pd.DataFrame) -> None:
        ForecastAgent.validate_weather(weather)
        required = set(COLUMNS)
        missing = required - set(pred.columns)
        if missing:
            raise ValueError(f"Prediction is missing columns: {sorted(missing)}")
        if len(pred) != len(weather):
            raise ValueError("Prediction and weather lengths differ")
        if not pd.DatetimeIndex(pred["time"]).equals(pd.DatetimeIndex(weather["time"])):
            raise ValueError("Prediction timestamps differ from weather")
        if not np.isfinite(pred[COLUMNS[1:]].to_numpy(dtype=float)).all():
            raise ValueError("Prediction components must be finite")
        if not np.array_equal(pred["lead_hours"], np.arange(1, 49)):
            raise ValueError("lead_hours must be 1..48")
        if not np.allclose(pred["wind_fc"], weather["wind_fc"], rtol=0, atol=1e-8):
            raise ValueError("Prediction wind differs from weather")
        values = pred["prediction"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Prediction must be finite and in [0, 1]")
        zero_wind = pred["wind_fc"].to_numpy(dtype=float) <= 0
        if (zero_wind & (values > 0.05)).any():
            raise ValueError("Prediction is too high when forecast wind is zero")
        expected = np.clip(pred["level1"] + pred["residual_pred"], 0, 1)
        if not np.allclose(values, expected, rtol=0, atol=1e-8):
            raise ValueError("prediction does not match level1 + residual_pred")
        if (np.abs(values - pred["level1"].to_numpy()) > 0.3).any():
            LOG.warning("Large deviation from level1 (>0.3 normalized power)")

    def save_forecast(self, turbine_id: int, issue_time: str | pd.Timestamp, pred: pd.DataFrame) -> Path:
        stamp = issue_timestamp(issue_time)
        if turbine_id not in (1, 2):
            raise ValueError("Unsupported turbine_id")
        # Validate even when called as an independent tool.
        weather = pred[["time", "wind_fc"]].copy()
        weather["temperature"] = 0.0
        self.validate_prediction(pred, weather)
        expected = pd.date_range(stamp + pd.Timedelta(hours=1), periods=48, freq="h")
        if not pd.DatetimeIndex(pred["time"]).equals(expected):
            raise ValueError("Prediction timestamps differ from issue_time")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        path = self.output_dir / f"turbine_{turbine_id}_{stamp.strftime('%Y%m%dT%H%M%SZ')}_{run_id}.json"
        records = json.loads(pred[COLUMNS].to_json(orient="records", date_format="iso"))
        payload = {"turbine_id": turbine_id, "issue_time": stamp.isoformat(),
                   "run_id": run_id, "weather": pred.attrs.get("weather", {}),
                   "model": pred.attrs.get("model", {}), "forecast": records}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
        self.last_saved_path = path
        LOG.info("saved turbine=%s rows=48 path=%s", turbine_id, path)
        return path

    def forecast(self, turbine_id: int, issue_time: str | pd.Timestamp) -> pd.DataFrame:
        self.last_saved_path = None
        self._model_metadata = {}
        stamp = issue_timestamp(issue_time)
        LOG.info("loading_weather turbine=%s issue=%s", turbine_id, stamp)
        weather = self._weather(turbine_id, stamp)
        LOG.info("running_model turbine=%s", turbine_id)
        level1, residual, prediction = self._predict(turbine_id, stamp, weather)
        result = weather[["time", "wind_fc"]].copy()
        result["level1"] = level1
        result["residual_pred"] = residual
        result["prediction"] = prediction
        result["lead_hours"] = np.arange(1, len(result) + 1)
        self.validate_prediction(result, weather)
        result = result[COLUMNS]
        result.attrs = {"weather": {k: v for k, v in weather.attrs.items() if not k.startswith("_")},
                        "model": dict(self._model_metadata)}
        self.save_forecast(turbine_id, stamp, result)
        return result

    def run(self, turbine_id: int, issue_time: str | pd.Timestamp, *, max_attempts: int = 2) -> pd.DataFrame:
        """Retry transient I/O only; each attempt reloads weather and artifacts.

        Call again when inputs update; each successful run saves a new version.
        Validation/model errors are not repaired by changing or inventing inputs.
        """
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        for attempt in range(max_attempts):
            try:
                return self.forecast(turbine_id, issue_time)
            except (FileNotFoundError, ValueError):
                LOG.exception("forecast_failed turbine=%s", turbine_id)
                raise
            except (TimeoutError, ConnectionError, RequestsTimeout, RequestsConnectionError):
                LOG.warning("transient_failure attempt=%s", attempt + 1)
                if attempt + 1 == max_attempts:
                    raise
        raise RuntimeError("Unreachable")
