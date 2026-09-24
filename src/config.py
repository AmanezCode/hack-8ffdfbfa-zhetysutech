from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = Path(os.environ.get("WIND_DATA_DIR", ROOT / "data" / "raw"))
DATA_CACHE = ROOT / "data" / "cache"
ARTIFACTS = ROOT / "artifacts"
FORECASTS = ROOT / "forecasts"

TURBINES = {
    1: {"lat": 43.645150, "lon": 78.535604, "csv": "turbine1.csv"},
    2: {"lat": 43.643198, "lon": 78.538828, "csv": "turbine2.csv"},
}

ELEVATION_M = 555.0

# All timestamps inside the pipeline are naive UTC. The SCADA CSV is on a fixed
# UTC+6 clock for the whole history: cross-correlation with NWP wind peaks at
# +6 h both before and after Kazakhstan moved to UTC+5 on 2024-03-01
# (scripts/check_time_alignment.py). Forecasts are published on the same clock so
# they line up with the hidden February actuals.
SCADA_UTC_OFFSET_HOURS = 6
ISSUE_HOUR_UTC = 17  # 23:00 on the SCADA clock: issued at the end of the day for the next two days
# The agent re-checks every 6 h for fresher weather runs, so the model is trained
# on issues at all four update hours; CV is scored on the official 17:00 UTC issues.
UPDATE_INTERVAL_HOURS = 6
TRAIN_ISSUE_HOURS_UTC = (17, 23, 5, 11)
MIN_READINGS_PER_HOUR = 4  # of 6 ten-minute readings; sparser hours are not used as labels

HISTORY_START = "2023-03-11"
HISTORY_END = "2026-01-31"
TEST_START = "2026-02-01"
TEST_END = "2026-02-28"

# Previous-run archive (forecasts as issued 1..3 days before valid time) begins
# 2024-02-16; training cannot start earlier without using weather unavailable at issue time.
WEATHER_START = "2024-02-16"
WEATHER_END = "2026-03-02"
PREVIOUS_RUN_DAYS = 3
# Global models start a run every 6 h (00/06/12/18 UTC). Init-to-availability on
# Open-Meteo ranges from 3.8 h (ICON) to 8.1 h (GFS 0.25, ECMWF IFS) and 9.5 h
# (JMA); best_match may use any of them, so a run counts as published 10 h after
# its start (scripts/check_publication_delay.py). A weather value is used only if
# the run that produced it was published by the issue time; for the 17/23/05/11 UTC
# issues every run used started at least 11 h before issue.
RUN_CYCLE_HOURS = 6
PUBLICATION_DELAY_HOURS = 10

HORIZON_HOURS = 48
