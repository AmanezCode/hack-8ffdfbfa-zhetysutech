"""Adapt aware UTC agent inputs to the team's naive UTC Previous Runs API."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import model
from src.config import ARTIFACTS, PUBLICATION_DELAY_HOURS, TURBINES
from src.data import load_turbine_hourly
from src.features import FEATURE_COLUMNS
from src.weather import load_weather, weather_snapshot, weather_provenance


def load_history(turbine_id: int) -> pd.DataFrame:
    return load_turbine_hourly(turbine_id)


def load_team_weather(lat: float, lon: float, issue_time: pd.Timestamp) -> pd.DataFrame:
    matches = [t for t, c in TURBINES.items() if abs(c["lat"] - lat) < 1e-6 and abs(c["lon"] - lon) < 1e-6]
    if len(matches) != 1:
        raise ValueError("Coordinates must match a configured turbine for Previous Runs")
    turbine_id = matches[0]
    issue = issue_time.tz_convert("UTC").tz_localize(None)
    times = pd.date_range(issue, periods=50, freq="h")
    archive = load_weather(turbine_id).reindex(times)
    snapshot = weather_snapshot(archive, issue, times[1:-1])
    missing = snapshot.index[snapshot[["wind_fc", "temp_fc"]].isna().any(axis=1)]
    if len(missing):
        raise model.IncompleteWeatherError(missing)
    frame = snapshot.rename(columns={"temp_fc": "temperature"}).reset_index()
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize("UTC")
    provenance = weather_provenance(turbine_id)
    frame.attrs.update(source="Open-Meteo Previous Runs", provenance=provenance,
                       publication_delay_hours=PUBLICATION_DELAY_HOURS,
                       availability_rule="run_day * 24 >= lead_hours + publication_delay_hours",
                       publication_delay_is_historical_assumption=True,
                       run_day=frame["run_day"].astype(int).tolist(),
                       _team_archive=archive, turbine_id=turbine_id)
    return frame


def run_team_model(turbine_id: int, issue_time: pd.Timestamp, weather: pd.DataFrame,
                   artifacts_dir: str | Path = ARTIFACTS, *, history: pd.DataFrame | None = None):
    directory = Path(artifacts_dir)
    if directory.resolve() == ARTIFACTS.resolve():
        artifacts = model.load_artifacts(turbine_id)
    else:
        artifacts = joblib.load(directory / f"model_t{turbine_id}.joblib")
        booster_path = directory / f"booster_t{turbine_id}.txt"
        artifacts["booster"] = lgb.Booster(model_str=booster_path.read_text(encoding="utf-8")) if booster_path.exists() else None
        artifacts["manifest"] = json.loads((directory / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))
    manifest = artifacts["manifest"]
    if manifest["feature_schema"] != FEATURE_COLUMNS or manifest["variant"] != artifacts["variant"]:
        raise ValueError("Model manifest does not match current feature/variant contract")
    if manifest["turbine_id"] != turbine_id:
        raise ValueError("Model manifest turbine mismatch")
    if artifacts["variant"] != "physics" and artifacts.get("booster") is None:
        raise ValueError("Selected ML variant requires booster_tN.txt")
    if artifacts.get("booster") is not None:
        if artifacts["booster"].feature_name() != FEATURE_COLUMNS:
            raise ValueError("Booster feature order differs from current ML schema")
        artifacts["booster"].params["num_threads"] = 1
    issue = issue_time.tz_convert("UTC").tz_localize(None)
    if pd.Timestamp(manifest["train_target_end"]) + pd.Timedelta(hours=1) > issue:
        raise ValueError("Model trained on labels unavailable at issue_time")
    archive = weather.attrs.get("_team_archive")
    if not isinstance(archive, pd.DataFrame):
        raise ValueError("Team ML requires the Previous Runs loader, not a flat vintage JSON")
    if weather.attrs.get("turbine_id") != turbine_id:
        raise ValueError("Weather turbine mismatch")
    history = load_history(turbine_id) if history is None else history.copy()
    if history.index.tz is not None:
        history.index = history.index.tz_convert("UTC").tz_localize(None)
    history = history.sort_index().loc[:issue - pd.Timedelta(hours=1)]
    result = model.run_forecast_model(history, archive, issue, artifacts)
    if result is None or len(result) != 48:
        raise ValueError("Team ML did not return a complete 48-hour forecast")
    times = result.index.tz_localize("UTC")
    if not times.equals(pd.DatetimeIndex(weather["time"])):
        raise ValueError("Team ML timestamps differ from requested horizon")
    if not np.allclose(result["wind_fc"], weather["wind_fc"], rtol=0, atol=1e-8):
        raise ValueError("Team ML weather alignment failed")
    if not np.array_equal(result["lead_hours"], np.arange(1, 49)):
        raise ValueError("Team ML lead hours differ from contract")
    if not np.array_equal(result["run_day"], weather["run_day"]):
        raise ValueError("Team ML run selection differs from validated weather")
    result.attrs["model"] = {"model_version": manifest["model_version"], "variant": artifacts["variant"],
                             "train_target_end": manifest["train_target_end"]}
    return result
