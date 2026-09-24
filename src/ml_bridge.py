"""Adapt aware UTC agent inputs to the team's naive UTC Previous Runs API."""
import functools
from pathlib import Path

import numpy as np
import pandas as pd

from src import model
from src.config import ARTIFACTS, PUBLICATION_DELAY_HOURS, TURBINES
from src.data import load_turbine_hourly
from src.features import FEATURE_COLUMNS, hour_means
from src.weather import fetch_previous_runs, load_weather, weather_snapshot, weather_provenance


@functools.lru_cache(maxsize=len(TURBINES))
def load_history(turbine_id: int) -> pd.DataFrame:
    """Parsed once per process: the SCADA CSV dominates the cost of a forecast otherwise."""
    return load_turbine_hourly(turbine_id)


def turbine_for(lat: float, lon: float) -> int:
    matches = [t for t, c in TURBINES.items() if abs(c["lat"] - lat) < 1e-6 and abs(c["lon"] - lon) < 1e-6]
    if len(matches) != 1:
        raise ValueError("Coordinates must match a configured turbine for Previous Runs")
    return matches[0]


def load_team_weather(lat: float, lon: float, issue_time: pd.Timestamp, *, refresh: bool = False) -> pd.DataFrame:
    """Weather admissible at issue_time for the next 48 h.

    Reads the pinned Previous Runs archive. With refresh=True it also queries
    Open-Meteo live by the turbine's coordinates for the days around the issue and
    lets those values take precedence; the pinned archive is not overwritten.
    """
    turbine_id = turbine_for(lat, lon)
    issue = issue_time.tz_convert("UTC").tz_localize(None)
    # issue .. issue+50 h: the model reads one hour on each side of the 48 h horizon, plus one
    # more for the hour-interval means.
    times = pd.date_range(issue, periods=51, freq="h")
    archive = load_weather(turbine_id)
    live_meta = None
    if refresh:
        start = (issue - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (issue + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        live, live_meta = fetch_previous_runs(lat, lon, start, end)
        archive = live.combine_first(archive)
    archive = archive.reindex(times)

    # The same per-hour weather the model reads, so the agent validates what the model uses.
    snapshot = hour_means(weather_snapshot(archive, issue, times[1:50]))
    missing = snapshot.index[snapshot[["wind_fc", "temp_fc"]].isna().any(axis=1)]
    if len(missing):
        raise model.IncompleteWeatherError(missing)
    # The agent validates only the core series; extras may legitimately be NaN and
    # reach the model through the archive attached below.
    frame = snapshot[["run_day", "wind_fc", "temp_fc"]].rename(columns={"temp_fc": "temperature"}).reset_index()
    frame["time"] = pd.to_datetime(frame["time"]).dt.tz_localize("UTC")
    provenance = weather_provenance(turbine_id)
    frame.attrs.update(source="Open-Meteo Previous Runs", provenance=provenance,
                       publication_delay_hours=PUBLICATION_DELAY_HOURS,
                       availability_rule="run behind previous_dayN (latest 6-hourly start <= valid time - N*24 h) public by issue time",
                       publication_delay_is_historical_assumption=True,
                       run_day=frame["run_day"].astype(int).tolist(),
                       _team_archive=archive, turbine_id=turbine_id)
    if live_meta is not None:
        frame.attrs["live_refresh"] = {k: live_meta[k] for k in ("retrieved_at", "sha256", "grid_latitude", "grid_longitude")}
    return frame


def model_card(manifest: dict) -> dict:
    """What the agent needs from the manifest to analyse and log a forecast."""
    chosen = manifest["variant"]
    return {
        "model_version": manifest["model_version"],
        "variant": chosen,
        "train_target_end": manifest["train_target_end"],
        "cv_accuracy": manifest["cv_mean"][chosen].get("accuracy"),
        "cv_mae": manifest["cv_mean"][chosen]["MAE"],
        "expected_abs_error_by_lead": manifest.get("expected_abs_error_by_lead"),
        "deviation_q99": manifest.get("deviation_q99"),
        "cv_within_10pct": manifest["cv_mean"][chosen].get("within_10pct"),
        "interval_cv_coverage": (manifest.get("interval") or {}).get("cv_coverage"),
    }


def run_team_model(turbine_id: int, issue_time: pd.Timestamp, weather: pd.DataFrame,
                   artifacts_dir: str | Path = ARTIFACTS, *, history: pd.DataFrame | None = None):
    directory = Path(artifacts_dir)
    artifacts = model.load_artifacts(turbine_id, None if directory.resolve() == ARTIFACTS.resolve() else directory)
    manifest = artifacts["manifest"]
    if manifest["feature_schema"] != FEATURE_COLUMNS or manifest["variant"] != artifacts["variant"]:
        raise ValueError("Model manifest does not match current feature/variant contract")
    if manifest["turbine_id"] != turbine_id:
        raise ValueError("Model manifest turbine mismatch")
    if artifacts["variant"] != "physics" and artifacts.get("booster") is None:
        raise ValueError("Selected ML variant requires booster_tN.txt")
    if artifacts["variant"] == "combined" and (artifacts.get("wind_model") is None or artifacts.get("anemometer_curve") is None):
        raise ValueError("Combined variant requires booster_wind_tN.txt and the anemometer power curve")
    for name in ("booster", "wind_model"):
        if artifacts.get(name) is not None:
            if artifacts[name].feature_name() != FEATURE_COLUMNS:
                raise ValueError(f"{name} feature order differs from current ML schema")
            artifacts[name].set_threads(1)
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
    result.attrs["model"] = model_card(manifest)
    return result
