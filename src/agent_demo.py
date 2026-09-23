"""Synthetic integration demo, NOT a historical forecast or ML evaluation."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_agent import ForecastAgent, ROOT


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    issue = pd.Timestamp("2026-01-31T23:00:00+05:00")

    def synthetic_weather(lat, lon, issue_time):
        frame = pd.DataFrame({
            "time": pd.date_range(issue_time + pd.Timedelta(hours=1), periods=48, freq="h"),
            "wind_fc": np.linspace(3, 12, 48), "temperature": np.full(48, 5.0),
        })
        frame.attrs.update(source="SYNTHETIC_DEMO_NOT_HISTORICAL", latitude=lat, longitude=lon,
                           issued_at=issue_time.isoformat(), available_at=issue_time.isoformat())
        return frame

    def synthetic_model(turbine_id, issue_time, weather):
        level1 = np.clip((weather["wind_fc"].to_numpy() / 15) ** 3, 0, 1)
        residual = np.zeros(48)
        return level1, residual, level1.copy()

    output_root = ROOT / "forecasts" / "demo"
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="synthetic_", dir=output_root))
    agent = ForecastAgent(output_dir=directory, coordinates={1: (0.0, 0.0)},
                          weather_loader=synthetic_weather, model_runner=synthetic_model)
    result = agent.run(1, issue)
    print("SYNTHETIC DEMO ONLY - no trained ML or historical weather used")
    print(result.head().to_string(index=False))
    print(f"rows={len(result)} output={agent.last_saved_path}")


if __name__ == "__main__":
    main()
