"""Strict vintage reader, separate from the ML team's weather downloader."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "weather"
COORDINATE_TOLERANCE = 1e-6


def _time(value: object, name: str) -> pd.Timestamp:
    if not isinstance(value, (str, pd.Timestamp)):
        raise ValueError(f"{name} requires an aware timestamp")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("missing timezone")
        return stamp.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} requires an aware timestamp") from exc


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} requires a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} requires a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} requires a finite number")
    return result


def _coordinates(lat: object, lon: object) -> tuple[float, float]:
    lat, lon = _number(lat, "latitude"), _number(lon, "longitude")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coordinates out of range")
    return lat, lon


def _load(path: Path, lat: float, lon: float, issue: pd.Timestamp) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"source", "issued_at", "available_at", "latitude", "longitude", "wind_unit", "hourly"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Complete vintage provenance envelope required")
    if not isinstance(payload["source"], str) or not payload["source"].strip():
        raise ValueError("Nonempty source required")
    issued = _time(payload["issued_at"], "issued_at")
    available = _time(payload["available_at"], "available_at")
    if issued > issue or available > issue or available < issued:
        raise ValueError("Require issued_at <= available_at <= issue_time")
    latitude, longitude = _coordinates(payload["latitude"], payload["longitude"])
    if abs(latitude - lat) > COORDINATE_TOLERANCE or abs(longitude - lon) > COORDINATE_TOLERANCE:
        raise ValueError("Vintage coordinates do not match")
    unit = payload["wind_unit"]
    if unit not in ("m/s", "km/h"):
        raise ValueError("wind_unit must be m/s or km/h")
    hourly = payload["hourly"]
    if not isinstance(hourly, dict):
        raise ValueError("hourly must be an object")
    for key in ("time", "wind_fc", "temperature"):
        if not isinstance(hourly.get(key), list) or len(hourly[key]) not in (48, 50):
            raise ValueError(f"hourly.{key} requires 48 values, or 50 including ML boundary hours")
    if len({len(hourly[key]) for key in ("time", "wind_fc", "temperature")}) != 1:
        raise ValueError("Hourly arrays must have equal lengths")
    times = pd.DatetimeIndex([_time(value, "hourly.time") for value in hourly["time"]])
    padded = len(times) == 50
    expected = pd.date_range(issue if padded else issue + pd.Timedelta(hours=1), periods=len(times), freq="h")
    if not times.equals(expected):
        raise ValueError("Require ordered hourly times issue_time + 1 through + 48 UTC")
    wind = [_number(value, "wind_fc") for value in hourly["wind_fc"]]
    if any(value < 0 for value in wind):
        raise ValueError("Negative wind is invalid")
    temperature = [_number(value, "temperature") for value in hourly["temperature"]]
    frame = pd.DataFrame({"time": times, "wind_fc": [v / 3.6 for v in wind] if unit == "km/h" else wind,
                          "temperature": temperature})
    frame.attrs.update(source=payload["source"], issued_at=issued, available_at=available,
                       latitude=latitude, longitude=longitude, issue_time=issue,
                       wind_unit="m/s", original_wind_unit=unit, archive_path=str(path.resolve()),
                       provenance_verified=False)
    if padded:
        padding = frame.iloc[[0, -1]].to_dict(orient="records")
        frame = frame.iloc[1:-1].reset_index(drop=True)
        frame.attrs["boundary_weather"] = padding
    return frame


def load_weather(
    lat: float,
    lon: float,
    issue_date: str | pd.Timestamp,
    archive_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a strict local vintage and return 48 hours after aware issue_date.

    JSON requires source, issued_at, available_at, latitude, longitude, wind_unit
    ('m/s' or 'km/h'), and hourly={time, wind_fc, temperature} with 48 values each.
    The team's ML requires 50 input hours (issue+0..49); two boundary hours are
    retained in attrs.boundary_weather while output still contains +1..48.
    All times must be timezone-aware. Output time is UTC, wind is m/s, and
    temperature is supplied in Celsius. Coordinate tolerance is 1e-6 absolute
    degrees. Metadata is exposed in DataFrame.attrs. Source provenance is a
    supplier declaration and cannot be cryptographically verified.

    By default search data/weather/*.json, skipping invalid/ineligible files,
    selecting latest issued_at, then available_at, then filename. No eligible
    vintage raises ValueError. Explicit files are validated without fallback;
    unreadable explicit files raise OSError. No external fetch or implicit use
    of data/raw/forecast.json is performed.
    """
    lat, lon = _coordinates(lat, lon)
    issue = _time(issue_date, "issue_date")
    if archive_path is not None:
        return _load(Path(archive_path), lat, lon, issue)
    candidates = []
    errors = []
    for path in sorted(DEFAULT_ARCHIVE.glob("*.json")):
        try:
            candidates.append(_load(path, lat, lon, issue))
        except (ValueError, OSError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not candidates:
        detail = "; ".join(errors) or "no JSON archives found"
        raise ValueError(f"No eligible weather vintage in {DEFAULT_ARCHIVE}: {detail}")
    return max(candidates, key=lambda f: (f.attrs["issued_at"], f.attrs["available_at"], f.attrs["archive_path"]))
