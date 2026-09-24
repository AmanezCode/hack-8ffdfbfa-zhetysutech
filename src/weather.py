"""Weather inputs with provable availability at issue time.

Source: Open-Meteo Previous Runs API. For every valid hour it archives the value
from the run issued 0, 1, 2, ... days earlier (``*_previous_dayN``: NWP lead
time N*24..N*24+23 h). The plain Historical Forecast API is not used: it
stitches the first hours of successive runs, so a 48 h forecast built from it
would contain runs published after the issue time.

The core series (wind, temperature and a few extra variables) comes from one
model, ICON, over the whole history. Open-Meteo's blended ``best_match`` is not
used for it: here it equals ICON until 2025-09-30 and ECMWF IFS 9 km from
2025-10-01, so a model trained on it would be scored and used on a different
model than it learned from. The archive also holds 100 m wind from four more
models (ECMWF IFS 0.25, ECMWF IFS 9 km, ECMWF AIFS, GFS) and 10 m wind from three
(UK Met Office, CMA, JMA): their disagreement is a strong signal of forecast
error. All of them publish within 10 h of initialisation, inside the
availability rule.
"""

import gzip
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.config import DATA_CACHE, HISTORY_START, PREVIOUS_RUN_DAYS, PUBLICATION_DELAY_HOURS, RUN_CYCLE_HOURS, TURBINES, WEATHER_END

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
CORE = {"wind_speed_100m": "wind", "temperature_2m": "temp"}
EXTRA = {"wind_speed_10m": "wind10", "wind_gusts_10m": "gust", "wind_direction_100m": "wdir",
         "surface_pressure": "pres", "relative_humidity_2m": "rh"}
CORE_MODEL = "icon_seamless"
ENSEMBLE = {"ecmwf_ifs025": "ecmwf", "ecmwf_ifs": "ifs9", "gfs_seamless": "gfs", "ecmwf_aifs025_single": "aifs"}
ENSEMBLE_VARIABLE = "wind_speed_100m"
# Models without 100 m wind in the archive contribute their 10 m wind.
SURFACE_ENSEMBLE = {"ukmo_seamless": "ukmo", "cma_grapes_global": "cma", "jma_seamless": "jma"}
SURFACE_VARIABLE = "wind_speed_10m"
CACHE_VERSION = "v4"


def day_column(base: str, day: int, model: str = "") -> str:
    return f"{base}_previous_day{day}" + (f"_{model}" if model else "")


def _day_columns(base: str, with_day0: bool = False) -> list[str]:
    return ([base] if with_day0 else []) + [day_column(base, n) for n in range(1, PREVIOUS_RUN_DAYS + 1)]


def _get(params: dict) -> tuple[pd.DataFrame, dict, bytes]:
    response = requests.get(ENDPOINT, params=params, timeout=300)
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time").sort_index(), payload, response.content


def fetch_previous_runs(lat: float, lon: float, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict]:
    common = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
              "wind_speed_unit": "ms", "timezone": "GMT"}
    core_columns = [c for base in CORE for c in _day_columns(base, with_day0=True)]
    extra_columns = [c for base in EXTRA for c in _day_columns(base)]
    params = common | {"hourly": ",".join(core_columns + extra_columns), "models": CORE_MODEL}
    frame, payload, content = _get(params)
    digest = hashlib.sha256(content)

    ensemble_meta = {}
    members = [(model, short, ENSEMBLE_VARIABLE) for model, short in ENSEMBLE.items()]
    members += [(model, short, SURFACE_VARIABLE) for model, short in SURFACE_ENSEMBLE.items()]
    for model, short, variable in members:
        columns = _day_columns(variable)
        part, part_payload, part_content = _get(common | {"hourly": ",".join(columns), "models": model})
        frame = frame.join(part.rename(columns={c: f"{c}_{short}" for c in columns}))
        digest.update(part_content)
        ensemble_meta[short] = {"model": model, "grid": [part_payload["latitude"], part_payload["longitude"]],
                                "sha256": hashlib.sha256(part_content).hexdigest()}

    meta = {
        "provider": "open-meteo",
        "endpoint": ENDPOINT,
        "params": params,
        "grid_latitude": payload["latitude"],
        "grid_longitude": payload["longitude"],
        "grid_elevation": payload["elevation"],
        "units": payload["hourly_units"],
        "core_model": CORE_MODEL,
        "ensemble": ensemble_meta,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest.hexdigest(),
    }
    return frame, meta


def _cache_stem(turbine_id: int, start_date: str, end_date: str):
    return DATA_CACHE / f"previous_runs_{CACHE_VERSION}_t{turbine_id}_{start_date}_{end_date}_d{PREVIOUS_RUN_DAYS}"


def load_weather(turbine_id: int, start_date: str = HISTORY_START, end_date: str = WEATHER_END, refresh: bool = False) -> pd.DataFrame:
    """Cached previous-run archive, hourly, naive UTC index. Provenance sits next to the data."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(turbine_id, start_date, end_date)
    data_path, meta_path = stem.with_suffix(".csv.gz"), stem.with_suffix(".json")

    if data_path.exists() and meta_path.exists() and not refresh:
        return pd.read_csv(data_path, parse_dates=["time"]).set_index("time")

    turbine = TURBINES[turbine_id]
    frame, meta = fetch_previous_runs(turbine["lat"], turbine["lon"], start_date, end_date)
    # mtime=0 keeps the compressed bytes identical for identical data.
    with gzip.GzipFile(data_path, "wb", mtime=0) as handle:
        handle.write(frame.to_csv().encode("utf-8"))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return frame


def weather_provenance(turbine_id: int, start_date: str = HISTORY_START, end_date: str = WEATHER_END) -> dict:
    return json.loads(_cache_stem(turbine_id, start_date, end_date).with_suffix(".json").read_text(encoding="utf-8"))


def run_start(issue_time: pd.Timestamp, lead_hours, run_day) -> np.ndarray:
    """Start of the run behind ``previous_dayN`` for valid time issue_time + lead.

    That value comes from the latest run started at or before valid time - N*24 h;
    runs start on the RUN_CYCLE_HOURS grid (00/06/12/18 UTC).
    """
    lead = np.asarray(lead_hours, dtype=float)
    latest = issue_time + pd.to_timedelta(lead - 24.0 * np.asarray(run_day, dtype=float), unit="h")
    return pd.DatetimeIndex(latest).floor(f"{RUN_CYCLE_HOURS}h").to_numpy()


def run_published_by(issue_time: pd.Timestamp, lead_hours, run_day) -> np.ndarray:
    """True where the run behind ``previous_dayN`` was public at issue_time."""
    published = run_start(issue_time, lead_hours, run_day) + np.timedelta64(PUBLICATION_DELAY_HOURS, "h")
    return published <= np.datetime64(issue_time)


def run_day_for_lead(lead_hours, issue_time: pd.Timestamp) -> np.ndarray:
    """Freshest previous-run day whose run was published before the issue time."""
    lead = np.asarray(lead_hours, dtype=float)
    days = np.full(lead.shape, PREVIOUS_RUN_DAYS + 1)
    for day in range(PREVIOUS_RUN_DAYS, 0, -1):
        days = np.where(run_published_by(issue_time, lead, day), day, days)
    if days.max() > PREVIOUS_RUN_DAYS:
        raise ValueError(f"lead {lead.max():.0f}h has no run published before {issue_time} in {PREVIOUS_RUN_DAYS} days of archive")
    return days


def _stack(rows: pd.DataFrame, base: str, model: str = "") -> np.ndarray:
    columns = [day_column(base, n, model) for n in range(1, PREVIOUS_RUN_DAYS + 1)]
    return np.column_stack([rows[c].to_numpy(dtype=float) if c in rows else np.full(len(rows), np.nan) for c in columns])


def weather_snapshot(weather: pd.DataFrame, issue_time: pd.Timestamp, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather for ``times`` exactly as it could be known at ``issue_time``.

    The core series (ICON wind and temperature) picks the freshest admissible
    run per hour; if that run is missing in the archive it falls back to an older
    one (published even earlier, so still admissible). Extra variables and the
    per-model winds are read from the same run and stay NaN where that run lacks
    them. ``run_day`` records which run was used; hours with no admissible core
    run stay NaN.
    """
    lead = (times - issue_time) / pd.Timedelta(hours=1)
    required = run_day_for_lead(lead, issue_time)
    rows = weather.reindex(times)

    core = {short: _stack(rows, base) for base, short in CORE.items()}
    usable = np.logical_and.reduce([np.isfinite(stack) for stack in core.values()])
    usable &= np.arange(1, PREVIOUS_RUN_DAYS + 1)[None, :] >= required[:, None]
    found = usable.any(axis=1)
    chosen = np.where(found, usable.argmax(axis=1), 0)
    picked = np.arange(len(times))

    def at_chosen(stack: np.ndarray) -> np.ndarray:
        return np.where(found, stack[picked, chosen], np.nan)

    snapshot = pd.DataFrame(index=times)
    snapshot.index.name = "time"
    snapshot["run_day"] = np.where(found, chosen + 1, np.nan)
    for short, stack in core.items():
        snapshot[f"{short}_fc"] = at_chosen(stack)
    for base, short in EXTRA.items():
        snapshot[f"{short}_fc"] = at_chosen(_stack(rows, base))
    for short in ENSEMBLE.values():
        snapshot[f"wind_{short}_fc"] = at_chosen(_stack(rows, ENSEMBLE_VARIABLE, short))
    for short in SURFACE_ENSEMBLE.values():
        snapshot[f"wind10_{short}_fc"] = at_chosen(_stack(rows, SURFACE_VARIABLE, short))
    return snapshot
