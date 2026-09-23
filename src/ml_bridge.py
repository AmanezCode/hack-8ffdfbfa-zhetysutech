"""Connect the agent's UTC contract to the team's local-time physics/LightGBM API."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import model
from src.config import ARTIFACTS, DATA_RAW, TIMEZONE, TURBINES


def load_history(turbine_id: int) -> pd.DataFrame:
    path = DATA_RAW / TURBINES[turbine_id]["csv"]
    if not path.is_file():
        path = DATA_RAW / f"turbine_{turbine_id}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame.columns) != 5:
        raise ValueError("SCADA input requires five columns")
    frame.columns = ["id", "time", "wind_speed", "power", "temperature"]
    frame["time"] = pd.to_datetime(frame["time"], format="mixed", errors="raise")
    # Preserve missing hours; interpolation could incorporate future measurements.
    return frame.set_index("time").sort_index()[["wind_speed", "power", "temperature"]].resample("h").mean()


def run_team_model(turbine_id: int, issue_time: pd.Timestamp, weather: pd.DataFrame,
                   artifacts_dir: str | Path = ARTIFACTS, *, history: pd.DataFrame | None = None):
    directory = Path(artifacts_dir)
    if directory.resolve() == ARTIFACTS.resolve():
        artifacts = model.load_artifacts(turbine_id)
    else:
        bundle = joblib.load(directory / f"model_t{turbine_id}.joblib")
        artifacts = {"power_curve": bundle["power_curve"], "blend_weight": bundle["blend_weight"],
                     "booster": lgb.Booster(model_file=str(directory / f"residual_t{turbine_id}.txt"))}
    boundary = weather.attrs.get("boundary_weather")
    if not boundary or len(boundary) != 2:
        raise ValueError("Team ML requires archived weather for issue+0..49 (50 hours including boundaries)")
    padded = pd.concat([weather[["time", "wind_fc", "temperature"]], pd.DataFrame(boundary)], ignore_index=True)
    padded["time"] = pd.to_datetime(padded["time"], utc=True)
    padded = padded.set_index("time").sort_index()
    expected = pd.date_range(issue_time, periods=50, freq="h")
    if not padded.index.equals(expected) or not np.isfinite(padded.to_numpy(dtype=float)).all():
        raise ValueError("ML weather boundaries must complete 50 consecutive finite hours")
    padded.index = padded.index.tz_convert(TIMEZONE).tz_localize(None)
    padded = padded.rename(columns={"temperature": "temp_fc"})
    local_issue = issue_time.tz_convert(TIMEZONE).tz_localize(None)
    history = load_history(turbine_id) if history is None else history.copy()
    if history.index.tz is not None:
        history.index = history.index.tz_convert(TIMEZONE).tz_localize(None)
    history = history.sort_index().loc[:local_issue]
    result = model.run_forecast_model(history, padded, local_issue, artifacts)
    if result is None or len(result) != 48:
        raise ValueError("Team ML did not return a complete 48-hour forecast")
    times = result.index.tz_localize(TIMEZONE).tz_convert("UTC")
    if not times.equals(pd.DatetimeIndex(weather["time"])):
        raise ValueError("Team ML timestamps differ from requested horizon")
    if not np.allclose(result["wind_fc"], weather["wind_fc"], rtol=0, atol=1e-8):
        raise ValueError("Team ML weather alignment failed")
    if not np.array_equal(result["lead_hours"], np.arange(1, 49)):
        raise ValueError("Team ML lead hours differ from contract")
    return tuple(result[name].to_numpy(dtype=float) for name in ("level1", "residual_pred", "prediction"))
