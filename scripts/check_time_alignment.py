"""Which clock is the SCADA CSV on? Correlate 10-minute turbine wind with NWP
wind (UTC) under candidate offsets, per period around the 2024-03-01 switch of
Kazakhstan from UTC+6 to UTC+5."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_RAW, TURBINES  # noqa: E402
from src.weather import load_weather  # noqa: E402

PERIODS = [
    ("2023-03..2024-02", "2023-03-11", "2024-02-29"),
    ("2024-03..2024-12", "2024-03-02", "2024-12-31"),
    ("2025-01..2026-01", "2025-01-01", "2026-01-31"),
]
OFFSETS = [5.0, 5.5, 6.0, 6.5]


def main() -> None:
    nwp = load_weather(1)["wind_speed_100m"].resample("10min").interpolate()
    for turbine_id, turbine in TURBINES.items():
        raw = pd.read_csv(DATA_RAW / turbine["csv"], encoding="utf-8")
        wind = pd.Series(raw.iloc[:, 2].to_numpy(), index=pd.to_datetime(raw.iloc[:, 1]))
        for name, start, end in PERIODS:
            segment = wind.loc[start:end]
            scores = {}
            for offset in OFFSETS:
                shifted = segment.copy()
                shifted.index = shifted.index - pd.Timedelta(hours=offset)
                joined = pd.concat([shifted.rename("scada"), nwp.rename("nwp")], axis=1, join="inner").dropna()
                scores[offset] = joined["scada"].corr(joined["nwp"])
            best = max(scores, key=scores.get)
            cells = "  ".join(f"UTC+{o:g}: {c:.4f}" for o, c in scores.items())
            print(f"turbine {turbine_id} {name}: {cells}  -> best UTC+{best:g}")


if __name__ == "__main__":
    main()
