import re
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, TURBINES

RAW_COLUMNS = ["id", "time", "wind_speed", "power", "temperature"]
MAX_INTERPOLATE_HOURS = 3


def load_turbine_hourly(turbine_id: int) -> pd.DataFrame:
    """10-minute SCADA readings resampled to hourly means.

    Gaps up to MAX_INTERPOLATE_HOURS are interpolated; longer outages stay NaN
    so they can be dropped instead of fabricated (turbine 1 is offline for 41
    days in May-June 2024).
    """
    path = _find_turbine_csv(turbine_id)
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = RAW_COLUMNS
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()[["wind_speed", "power", "temperature"]]

    hourly = df.resample("1h").mean()
    full_index = pd.date_range(hourly.index[0], hourly.index[-1], freq="1h")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "time"
    return hourly.interpolate(limit=MAX_INTERPOLATE_HOURS, limit_area="inside")


def _find_turbine_csv(turbine_id: int) -> Path:
    """Find canonical files or the original HackAlem dataset filenames."""
    canonical = DATA_RAW / TURBINES[turbine_id]["csv"]
    if canonical.is_file():
        return canonical

    search_dirs = [DATA_RAW, Path.home() / "Desktop" / "track"]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for candidate in directory.glob("*.csv"):
            if re.search(rf"turbine\s*{turbine_id}\.csv$", candidate.name, re.IGNORECASE):
                return candidate

    raise FileNotFoundError(
        f"Не найден CSV для турбины {turbine_id}. Положите {canonical.name} в {DATA_RAW} "
        "или задайте WIND_DATA_DIR с папкой датасетов."
    )
