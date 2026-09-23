import numpy as np
import pandas as pd

from src.config import HORIZON_HOURS
from src.physics import air_density, density_corrected_wind

POWER_LAGS = [1, 2, 3, 6, 12, 24]
WIND_LAGS = [1, 3, 6]
POWER_ROLLS = [3, 6, 24]

# Only weather-driven features are used for prediction. Recent-actuals lags are
# still computed (baselines, validator, operational use when SCADA is live) but
# stay out of the model: during the February test period no actuals exist, and
# measurements showed the lags add nothing at a 24-48h horizon anyway.
FEATURE_COLUMNS = (
    ["wind_fc", "temp_fc", "air_density", "wind_corrected", "level1", "lead_hours", "hour", "day_of_week", "month"]
    + ["wind_fc_prev", "wind_fc_next", "wind_fc_mean3", "wind_fc_ramp", "wind_fc_std3", "level1_mean3"]
)

LAG_COLUMNS = (
    [f"power_lag_{lag}" for lag in POWER_LAGS]
    + [f"power_mean_{window}h" for window in POWER_ROLLS]
    + [f"wind_lag_{lag}" for lag in WIND_LAGS]
)


def issue_state(history: pd.DataFrame, issue_time: pd.Timestamp) -> dict | None:
    """Lag features frozen at the moment the forecast is issued.

    They stay constant across the whole horizon: a 48h-ahead row cannot use a
    1h-ago actual, and feeding predictions back as lags compounds error.
    """
    known = history.loc[:issue_time]
    if known.empty:
        return None

    state = {}
    for lag in POWER_LAGS:
        state[f"power_lag_{lag}"] = _value_at(known["power"], issue_time - pd.Timedelta(hours=lag - 1))
    for lag in WIND_LAGS:
        state[f"wind_lag_{lag}"] = _value_at(known["wind_speed"], issue_time - pd.Timedelta(hours=lag - 1))
    for window in POWER_ROLLS:
        state[f"power_mean_{window}h"] = known["power"].iloc[-window:].mean()

    if not np.isfinite(list(state.values())).all():
        return None
    return state


def build_features(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    issue_time: pd.Timestamp,
    power_curve,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame | None:
    state = issue_state(history, issue_time)
    target_times = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=horizon, freq="1h")
    # One hour of padding each side so ramp and rolling features exist at the edges.
    padded_times = pd.date_range(target_times[0] - pd.Timedelta(hours=1), target_times[-1] + pd.Timedelta(hours=1), freq="1h")
    padded = weather.reindex(padded_times)
    if padded["wind_fc"].isna().all():
        return None

    padded_corrected = density_corrected_wind(padded["wind_fc"], padded["temp_fc"])
    padded_level1 = pd.Series(power_curve.predict(padded_corrected), index=padded_times)
    padded_wind = padded["wind_fc"]

    frame = pd.DataFrame(index=target_times)
    frame.index.name = "time"
    frame["wind_fc"] = padded_wind.reindex(target_times).to_numpy()
    frame["temp_fc"] = padded["temp_fc"].reindex(target_times).to_numpy()
    frame["air_density"] = air_density(frame["temp_fc"])
    frame["wind_corrected"] = density_corrected_wind(frame["wind_fc"], frame["temp_fc"])
    frame["level1"] = power_curve.predict(frame["wind_corrected"])

    # A grid point 5 km from the turbine gets the timing of ramps wrong, so the
    # neighbourhood of the forecast carries signal the single-hour value does not.
    frame["wind_fc_prev"] = padded_wind.shift(1).reindex(target_times).to_numpy()
    frame["wind_fc_next"] = padded_wind.shift(-1).reindex(target_times).to_numpy()
    frame["wind_fc_mean3"] = padded_wind.rolling(3, center=True, min_periods=1).mean().reindex(target_times).to_numpy()
    frame["wind_fc_std3"] = padded_wind.rolling(3, center=True, min_periods=1).std().reindex(target_times).to_numpy()
    frame["wind_fc_ramp"] = frame["wind_fc"] - frame["wind_fc_prev"]
    frame["level1_mean3"] = padded_level1.rolling(3, center=True, min_periods=1).mean().reindex(target_times).to_numpy()

    frame["lead_hours"] = np.arange(1, horizon + 1)
    frame["hour"] = frame.index.hour
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month

    for key in LAG_COLUMNS:
        frame[key] = state[key] if state else np.nan

    return frame.dropna(subset=["wind_fc", "temp_fc"])[FEATURE_COLUMNS + LAG_COLUMNS]


def _value_at(series: pd.Series, timestamp: pd.Timestamp) -> float:
    if timestamp in series.index:
        return series.loc[timestamp]
    return np.nan
