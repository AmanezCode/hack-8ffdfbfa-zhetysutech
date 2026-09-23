"""Measured delay between NWP initialisation and availability on Open-Meteo,
per global model, from the metadata API. Justifies PUBLICATION_DELAY_HOURS:
Previous Runs uses ``best_match``, which may draw on any of these models, so
the delay must cover the slowest one with margin."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_CACHE, PUBLICATION_DELAY_HOURS  # noqa: E402

MODELS = ["dwd_icon", "ncep_gfs013", "ncep_gfs025", "cma_grapes_global", "ecmwf_ifs", "ecmwf_ifs025", "jma_gsm"]


def main() -> None:
    rows = []
    for model in MODELS:
        meta = requests.get(f"https://api.open-meteo.com/data/{model}/static/meta.json", timeout=30).json()
        init, available = meta["last_run_initialisation_time"], meta["last_run_availability_time"]
        rows.append(
            {
                "model": model,
                "run_initialisation_utc": datetime.fromtimestamp(init, timezone.utc).isoformat(timespec="minutes"),
                "run_availability_utc": datetime.fromtimestamp(available, timezone.utc).isoformat(timespec="minutes"),
                "delay_hours": round((available - init) / 3600, 2),
            }
        )

    slowest = max(row["delay_hours"] for row in rows)
    for row in rows:
        print(f"{row['model']:18s} {row['delay_hours']:5.2f} h")
    print(f"slowest {slowest:.2f} h, configured {PUBLICATION_DELAY_HOURS} h, margin {PUBLICATION_DELAY_HOURS - slowest:.2f} h")

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    report = {"retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "configured_hours": PUBLICATION_DELAY_HOURS, "models": rows}
    (DATA_CACHE / "model_publication_delays.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if slowest >= PUBLICATION_DELAY_HOURS:
        sys.exit(f"configured delay {PUBLICATION_DELAY_HOURS} h does not cover {slowest} h")


if __name__ == "__main__":
    main()
