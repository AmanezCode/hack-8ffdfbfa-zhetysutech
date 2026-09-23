from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
ARTIFACTS = ROOT / "artifacts"
FORECASTS = ROOT / "forecasts"

TURBINES = {
    1: {"lat": 43.645150, "lon": 78.535604, "csv": "turbine1.csv"},
    2: {"lat": 43.643198, "lon": 78.538828, "csv": "turbine2.csv"},
}

ELEVATION_M = 555.0
TIMEZONE = "Asia/Almaty"

HISTORY_START = "2023-03-11"
HISTORY_END = "2026-01-31"
TEST_START = "2026-02-01"
TEST_END = "2026-02-28"

HORIZON_HOURS = 48
VALIDATION_DAYS = 30
