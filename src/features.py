import numpy as np
import pandas as pd

from src.config import HORIZON_HOURS
from src.data import utc_to_scada
from src.physics import air_density, density_corrected_wind
from src.weather import weather_snapshot

POWER_LAGS = [1, 2, 3, 6, 12, 24]
WIND_LAGS = [1, 3, 6]
POWER_ROLLS = [3, 6, 24]

# Weather-driven only. SCADA lags are computed for baselines and the validator
# but kept out of the model: the February test has no actuals, and a replay
# must not depend on them.
FEATURE_COLUMNS = [
    "wind_fc", "temp_fc", "air_density", "wind_corrected", "level1", "level1_mean3",
    "run_day", "lead_hours", "hour", "day_of_week", "month",
    "wind_fc_prev", "wind_fc_next", "wind_fc_mean3", "wind_fc_std3", "wind_fc_ramp",
]

LAG_COLUMNS = (
    [f"power_lag_{lag}" for lag in POWER_LAGS]
    + [f"power_mean_{window}h" for window in POWER_ROLLS]
    + [f"wind_lag_{lag}" for lag in WIND_LAGS]
)


def issue_state(history: pd.DataFrame, issue_time: pd.Timestamp) -> dict:
    """SCADA values known at issue time: only hourly buckets that have fully closed.

    The bucket stamped H covers H:00..H:50, so at issue time T the latest
    complete one is T-1h. Missing values stay NaN.
    """
    known = history.loc[: issue_time - pd.Timedelta(hours=1)]
    state = {}
    for lag in POWER_LAGS:
        state[f"power_lag_{lag}"] = _value_at(known["power"], issue_time - pd.Timedelta(hours=lag))
    for lag in WIND_LAGS:
        state[f"wind_lag_{lag}"] = _value_at(known["wind_speed"], issue_time - pd.Timedelta(hours=lag))
    for window in POWER_ROLLS:
        state[f"power_mean_{window}h"] = known["power"].iloc[-window:].mean() if len(known) else np.nan
    return state


def build_features(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    issue_time: pd.Timestamp,
    power_curve=None,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """One row per target hour issue_time+1 .. issue_time+horizon (UTC).

    Rows whose weather is missing are kept with NaN so the caller can report
    them instead of silently returning a shorter forecast.
    """
    target_times = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=horizon, freq="1h", name="time")
    padded_times = pd.date_range(target_times[0] - pd.Timedelta(hours=1), target_times[-1] + pd.Timedelta(hours=1), freq="1h")

    # Neighbouring hours come from the same availability rule, so the centred
    # window never borrows a run published after issue_time.
    snapshot = weather_snapshot(weather, issue_time, padded_times)
    wind = snapshot["wind_fc"]
    corrected = pd.Series(density_corrected_wind(wind, snapshot["temp_fc"]), index=padded_times)

    frame = pd.DataFrame(index=target_times)
    frame["wind_fc"] = wind.reindex(target_times)
    frame["temp_fc"] = snapshot["temp_fc"].reindex(target_times)
    frame["run_day"] = snapshot["run_day"].reindex(target_times)
    frame["air_density"] = air_density(frame["temp_fc"])
    frame["wind_corrected"] = corrected.reindex(target_times)
    frame["wc_prev"] = corrected.shift(1).reindex(target_times)
    frame["wc_next"] = corrected.shift(-1).reindex(target_times)

    frame["wind_fc_prev"] = wind.shift(1).reindex(target_times)
    frame["wind_fc_next"] = wind.shift(-1).reindex(target_times)
    frame["wind_fc_mean3"] = wind.rolling(3, center=True, min_periods=1).mean().reindex(target_times)
    frame["wind_fc_std3"] = wind.rolling(3, center=True, min_periods=2).std().reindex(target_times)
    frame["wind_fc_ramp"] = frame["wind_fc"] - frame["wind_fc_prev"]

    scada_clock = utc_to_scada(target_times)
    frame["lead_hours"] = np.arange(1, horizon + 1)
    frame["hour"] = scada_clock.hour
    frame["day_of_week"] = scada_clock.dayofweek
    frame["month"] = scada_clock.month

    for key, value in issue_state(history, issue_time).items():
        frame[key] = value

    if power_curve is not None:
        add_level1(frame, power_curve)
    return frame


def add_level1(frame: pd.DataFrame, power_curve) -> pd.DataFrame:
    frame["level1"] = power_curve.predict(frame["wind_corrected"])
    neighbours = np.column_stack(
        [power_curve.predict(frame["wc_prev"]), frame["level1"], power_curve.predict(frame["wc_next"])]
    )
    finite = np.isfinite(np.column_stack([frame["wc_prev"], frame["wind_corrected"], frame["wc_next"]]))
    frame["level1_mean3"] = np.where(finite, neighbours, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
    frame.loc[~np.isfinite(frame["wind_corrected"]), ["level1", "level1_mean3"]] = np.nan
    return frame


def _value_at(series: pd.Series, timestamp: pd.Timestamp) -> float:
    return series.get(timestamp, np.nan)
