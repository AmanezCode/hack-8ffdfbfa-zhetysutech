"""Weather inputs with provable availability at issue time.

Source: Open-Meteo Previous Runs API. For every valid hour it archives the value
from the run issued 0, 1, 2, ... days earlier (``*_previous_dayN``: NWP lead
time N*24..N*24+23 h). The plain Historical Forecast API is not used: it
stitches the first hours of successive runs, so a 48 h forecast built from it
would contain runs published after the issue time.
"""

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.config import (
    DATA_CACHE,
    HISTORY_START,
    PREVIOUS_RUN_DAYS,
    PUBLICATION_DELAY_HOURS,
    TURBINES,
    WEATHER_END,
)

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
BASES = {"wind_speed_100m": "wind", "temperature_2m": "temp"}


def _columns() -> list[str]:
    suffixes = [""] + [f"_previous_day{n}" for n in range(1, PREVIOUS_RUN_DAYS + 1)]
    return [base + suffix for base in BASES for suffix in suffixes]


def fetch_previous_runs(lat: float, lon: float, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(_columns()),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    }
    response = requests.get(ENDPOINT, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()

    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time").sort_index()

    meta = {
        "provider": "open-meteo",
        "endpoint": ENDPOINT,
        "params": params,
        "grid_latitude": payload["latitude"],
        "grid_longitude": payload["longitude"],
        "grid_elevation": payload["elevation"],
        "units": payload["hourly_units"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "publication_delay_hours": PUBLICATION_DELAY_HOURS,
    }
    return frame, meta


def load_weather(turbine_id: int, start_date: str = HISTORY_START, end_date: str = WEATHER_END, refresh: bool = False) -> pd.DataFrame:
    """Cached previous-run archive, hourly, naive UTC index. Provenance sits next to the CSV."""
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    stem = DATA_CACHE / f"previous_runs_t{turbine_id}_{start_date}_{end_date}_d{PREVIOUS_RUN_DAYS}"
    csv_path, meta_path = stem.with_suffix(".csv"), stem.with_suffix(".json")

    if csv_path.exists() and meta_path.exists() and not refresh:
        return pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")

    turbine = TURBINES[turbine_id]
    frame, meta = fetch_previous_runs(turbine["lat"], turbine["lon"], start_date, end_date)
    frame.to_csv(csv_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return frame


def weather_provenance(turbine_id: int, start_date: str = HISTORY_START, end_date: str = WEATHER_END) -> dict:
    stem = DATA_CACHE / f"previous_runs_t{turbine_id}_{start_date}_{end_date}_d{PREVIOUS_RUN_DAYS}"
    return json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))


def run_day_for_lead(lead_hours) -> np.ndarray:
    """Freshest previous-run day guaranteed to be published before the issue time.

    A value from ``previous_dayN`` comes from a run initialised at most N*24 h
    before its valid time and published PUBLICATION_DELAY_HOURS later, so it is
    available at issue time when N*24 >= lead + delay.
    """
    lead = np.asarray(lead_hours, dtype=float)
    days = np.ceil((lead + PUBLICATION_DELAY_HOURS) / 24.0).astype(int)
    days = np.maximum(days, 1)
    if days.max() > PREVIOUS_RUN_DAYS:
        raise ValueError(f"lead {lead.max():.0f}h needs previous_day{days.max()}, archive has {PREVIOUS_RUN_DAYS}")
    return days


def weather_snapshot(weather: pd.DataFrame, issue_time: pd.Timestamp, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Weather for ``times`` exactly as it could be known at ``issue_time``.

    Uses the freshest admissible run per hour; if that run is missing in the
    archive, falls back to an older one (published even earlier, so still
    admissible). ``run_day`` records which one was used; hours with no
    admissible run stay NaN.
    """
    lead = (times - issue_time) / pd.Timedelta(hours=1)
    required = run_day_for_lead(lead)
    rows = weather.reindex(times)

    stacks = {
        short: np.column_stack([rows[f"{base}_previous_day{n}"].to_numpy() for n in range(1, PREVIOUS_RUN_DAYS + 1)])
        for base, short in BASES.items()
    }
    usable = np.logical_and.reduce([np.isfinite(stack) for stack in stacks.values()])
    usable &= np.arange(1, PREVIOUS_RUN_DAYS + 1)[None, :] >= required[:, None]

    found = usable.any(axis=1)
    chosen = np.where(found, usable.argmax(axis=1), 0)
    picked = np.arange(len(times))

    snapshot = pd.DataFrame(index=times)
    snapshot.index.name = "time"
    snapshot["run_day"] = np.where(found, chosen + 1, np.nan)
    for short, stack in stacks.items():
        snapshot[f"{short}_fc"] = np.where(found, stack[picked, chosen], np.nan)
    return snapshot
