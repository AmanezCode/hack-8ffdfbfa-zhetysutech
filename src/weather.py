import pandas as pd
import requests

from src.config import DATA_CACHE, TIMEZONE, TURBINES

ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
VARIABLES = "wind_speed_100m,temperature_2m"


def fetch_archived_forecast(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Weather forecast as it was archived for those dates, hourly, local time.

    wind_speed_unit=ms and timezone are explicit: the API defaults to km/h and
    UTC, which silently breaks both the power curve and the hour-of-day feature.
    """
    response = requests.get(
        ENDPOINT,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": VARIABLES,
            "wind_speed_unit": "ms",
            "timezone": TIMEZONE,
        },
        timeout=120,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "wind_fc": hourly["wind_speed_100m"],
            "temp_fc": hourly["temperature_2m"],
        }
    )
    return df.set_index("time").sort_index()


def load_weather(turbine_id: int, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_CACHE / f"weather_t{turbine_id}_{start_date}_{end_date}.csv"

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["time"]).set_index("time")

    turbine = TURBINES[turbine_id]
    df = fetch_archived_forecast(turbine["lat"], turbine["lon"], start_date, end_date)
    df.to_csv(cache_path)
    return df
