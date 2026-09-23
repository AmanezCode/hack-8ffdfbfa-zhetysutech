from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
TIME = "Статистическое время"
WIND = "Средняя скорость ветра(m/s)"
POWER = "Нормализованная активная мощность"
TEMP = "Средняя температура окружающей среды(°C)"


def load_turbines() -> pd.DataFrame:
    frames = []
    for turbine_id in (1, 2):
        path = RAW / f"turbine_{turbine_id}.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["timestamp"] = pd.to_datetime(frame[TIME], errors="coerce")
        frame = frame.dropna(subset=["timestamp", WIND, POWER, TEMP])
        frame["turbine"] = turbine_id
        frame["hour"] = frame["timestamp"].dt.floor("h")
        hourly = frame.groupby("hour", as_index=False).agg(
            wind=(WIND, "mean"), temperature=(TEMP, "mean"), target=(POWER, "mean"),
            samples=(POWER, "size"),
        )
        hourly["turbine"] = turbine_id
        frames.append(hourly)
    data = pd.concat(frames, ignore_index=True).sort_values(["hour", "turbine"])
    return data[data["samples"] >= 4].reset_index(drop=True)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["hour_of_day"] = out["hour"].dt.hour
    out["day_of_year"] = out["hour"].dt.dayofyear
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    for lag in (1, 2, 3, 24, 48, 168):
        if "target" in out:
            out[f"target_lag_{lag}"] = out.groupby("turbine")["target"].shift(lag)
        if "wind" in out:
            out[f"wind_lag_{lag}"] = out.groupby("turbine")["wind"].shift(lag)
    return out


FEATURES = ["turbine", "wind", "temperature", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    test = test.reset_index(drop=True)
    train_f = features(train).dropna(subset=FEATURES)
    # No target-derived feature is used here: future target values never enter the test matrix.
    test_f = features(test)
    test_f = test_f.dropna(subset=FEATURES)
    model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.2, random_state=42)
    model.fit(train_f[FEATURES], train_f["target"])
    pred = np.full(len(test), np.nan)
    pred[test_f.index.to_numpy()] = model.predict(test_f[FEATURES])
    return np.clip(pred, 0, 1)


def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-01-01")
    test_end = pd.Timestamp("2026-02-01")
    rows = []
    for origin in pd.date_range(cutoff, test_end - pd.Timedelta(hours=24), freq="168h"):
        train = data[data["hour"] < origin]
        test = data[(data["hour"] >= origin) & (data["hour"] < origin + pd.Timedelta(hours=24))]
        if len(test) < 40 or len(train) < 400:
            continue
        pred = fit_predict(train, test)
        valid = np.isfinite(pred)
        if valid.sum() == 0:
            continue
        rows.append({"origin": origin, "mae": mean_absolute_error(test.loc[valid, "target"], pred[valid]), "rmse": mean_squared_error(test.loc[valid, "target"], pred[valid]) ** 0.5, "n": int(valid.sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "backtest_results.csv", index=False)
    return result


def make_forecast(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    origin = data["hour"].max() + pd.Timedelta(hours=1)
    future = pd.MultiIndex.from_product([pd.date_range(origin, periods=horizon, freq="h"), [1, 2]], names=["hour", "turbine"]).to_frame(index=False)
    recent = data.groupby("turbine").tail(168)
    future = future.merge(recent.groupby("turbine", as_index=False).agg(wind=("wind", "mean"), temperature=("temperature", "mean")), on="turbine", how="left")
    pred = fit_predict(data, future)
    future["forecast_power"] = pred
    future.to_csv(OUT / "forecast.csv", index=False)
    return future


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=48)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data = load_turbines()
    summary = {"rows": int(len(data)), "start": str(data.hour.min()), "end": str(data.hour.max()), "turbines": sorted(data.turbine.unique().tolist())}
    (OUT / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    results = run_backtest(data)
    forecast = make_forecast(data, args.horizon)
    print(json.dumps({"audit": summary, "backtest_days": len(results), "mean_mae": float(results.mae.mean()) if len(results) else None, "mean_rmse": float(results.rmse.mean()) if len(results) else None, "forecast_rows": len(forecast)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
