"""Accuracy and efficiency summary from the trained artifacts.

Accuracy is 1 - MAE on normalised power (MAE as a share of rated capacity),
averaged over the rolling CV months, on the official daily 23:00 SCADA issues.
Skill = relative MAE reduction versus a reference forecast. Efficiency is
measured on this machine: training time from the manifest, then inference and
the full agent cycle timed here.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS, TURBINES  # noqa: E402
from src.data import load_turbine_hourly  # noqa: E402
from src.forecast_agent import ForecastAgent  # noqa: E402
from src.model import load_artifacts, run_forecast_model  # noqa: E402
from src.weather import load_weather  # noqa: E402

REFERENCES = {"physics": "power curve", "climatology": "climatology", "persistence": "persistence"}


def main() -> None:
    import logging
    logging.disable(logging.INFO)
    report = {}
    for turbine_id, cfg in TURBINES.items():
        manifest = json.loads((ARTIFACTS / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))
        chosen = manifest["variant"]
        cv = manifest["cv_mean"]
        folds = manifest["cv_folds"]
        by_lead = {label: sum(f["mae_by_lead"][label][chosen] for f in folds) / len(folds) for label in ("1-24h", "25-48h")}

        history, weather, artifacts = load_turbine_hourly(turbine_id), load_weather(turbine_id), load_artifacts(turbine_id)
        issues = pd.date_range("2026-01-31 17:00", "2026-02-27 17:00", freq="1D")
        started = time.perf_counter()
        for issue in issues:
            run_forecast_model(history, weather, issue, artifacts)
        inference_ms = (time.perf_counter() - started) / len(issues) * 1000

        with tempfile.TemporaryDirectory() as directory:
            agent = ForecastAgent(output_dir=directory, coordinates={turbine_id: (cfg["lat"], cfg["lon"])})
            started = time.perf_counter()
            for issue in issues:
                agent.run(turbine_id, issue.tz_localize("UTC"))
            agent_ms = (time.perf_counter() - started) / len(issues) * 1000

        report[turbine_id] = {
            "model": manifest["model_version"],
            "accuracy_1_minus_nMAE": round(cv[chosen]["accuracy"], 4),
            "nMAE": round(cv[chosen]["MAE"], 4),
            "nRMSE": round(cv[chosen]["RMSE"], 4),
            "bias": round(cv[chosen]["bias"], 4),
            "WAPE": round(cv[chosen]["WAPE"], 4),
            "daily_energy_error_share_of_capacity": round(cv[chosen]["daily_energy_error"], 4),
            "within_10pct_of_capacity": round(cv[chosen]["within_10pct"], 4),
            "interval_80_cv_coverage": round(manifest["interval"]["cv_coverage"], 4),
            "interval_80_mean_width": round(manifest["interval"]["cv_mean_width"], 4),
            "reissue_gain": {k: round(v["improvement"], 4) for k, v in manifest["update_gain"].items()},
            "nMAE_by_horizon": {k: round(v, 4) for k, v in by_lead.items()},
            "skill_vs": {name: round(1 - cv[chosen]["MAE"] / cv[ref]["MAE"], 4) for ref, name in REFERENCES.items()},
            "train_seconds": manifest["train_seconds"],
            "train_rows": manifest["train_rows"],
            "model_size_kb": round((ARTIFACTS / f"booster_t{turbine_id}.txt").stat().st_size / 1024, 1),
            "inference_ms_per_48h_forecast": round(inference_ms, 1),
            "agent_cycle_ms_per_forecast": round(agent_ms, 1),
        }

    for turbine_id, r in report.items():
        print(f"turbine {turbine_id} ({r['model']})")
        print(f"  accuracy (1 - nMAE)        {r['accuracy_1_minus_nMAE']:.1%}   nMAE {r['nMAE']:.1%}  nRMSE {r['nRMSE']:.1%}  bias {r['bias']:+.1%}")
        print(f"  by horizon                 1-24 h nMAE {r['nMAE_by_horizon']['1-24h']:.1%}, 25-48 h nMAE {r['nMAE_by_horizon']['25-48h']:.1%}")
        print(f"  within 10% of capacity     {r['within_10pct_of_capacity']:.1%} of hours")
        print(f"  daily energy error         {r['daily_energy_error_share_of_capacity']:.1%} of daily capacity;  WAPE {r['WAPE']:.1%}")
        print(f"  80% interval               CV coverage {r['interval_80_cv_coverage']:.1%}, mean width {r['interval_80_mean_width']:.1%}")
        print("  re-issue with fresher data " + ", ".join(f"{k} -{v:.1%} MAE" for k, v in r["reissue_gain"].items()))
        print("  skill (MAE reduction) vs   " + ", ".join(f"{k} {v:.1%}" for k, v in r["skill_vs"].items()))
        print(f"  efficiency                 train {r['train_seconds']:.0f} s on {r['train_rows']} rows, model {r['model_size_kb']} KB, "
              f"inference {r['inference_ms_per_48h_forecast']:.0f} ms, full agent cycle {r['agent_cycle_ms_per_forecast']:.0f} ms per 48 h forecast")
    (ARTIFACTS / "metrics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
