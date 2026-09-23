import pandas as pd

from src.config import DATA_RAW, MIN_READINGS_PER_HOUR, SCADA_UTC_OFFSET_HOURS, TURBINES

RAW_COLUMNS = ["id", "time", "wind_speed", "power", "temperature"]


def scada_to_utc(times):
    return times - pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def utc_to_scada(times):
    return times + pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def load_turbine_hourly(turbine_id: int) -> pd.DataFrame:
    """Hourly means of the 10-minute SCADA readings on a complete UTC hourly index.

    Hour H holds the readings stamped H:00..H:50 on the SCADA clock, the same
    bucketing the hourly target is defined on. Nothing is interpolated: hours
    with fewer than MIN_READINGS_PER_HOUR readings stay NaN, so outages never
    become fabricated labels.
    """
    path = DATA_RAW / TURBINES[turbine_id]["csv"]
    raw = pd.read_csv(path, encoding="utf-8")
    raw.columns = RAW_COLUMNS
    raw["time"] = scada_to_utc(pd.to_datetime(raw["time"]))
    raw = raw.set_index("time").sort_index()[["wind_speed", "power", "temperature"]]

    grouped = raw.resample("1h")
    hourly = grouped.mean()
    hourly["n_readings"] = grouped["power"].count()

    full_index = pd.date_range(hourly.index[0], hourly.index[-1], freq="1h", name="time")
    hourly = hourly.reindex(full_index)
    hourly["n_readings"] = hourly["n_readings"].fillna(0).astype(int)

    sparse = hourly["n_readings"] < MIN_READINGS_PER_HOUR
    hourly.loc[sparse, ["wind_speed", "power", "temperature"]] = float("nan")
    return hourly
