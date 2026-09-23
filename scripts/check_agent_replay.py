"""Replay both turbines through the agent without overwriting ML CSV forecasts."""
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict import issue_times
from src.config import FORECASTS, TURBINES
from src.forecast_agent import ForecastAgent


def main():
    logging.basicConfig(level=logging.ERROR)
    checks = []
    with tempfile.TemporaryDirectory(prefix="agent_replay_") as directory:
        for turbine, cfg in TURBINES.items():
            agent = ForecastAgent(output_dir=directory, coordinates={turbine: (cfg["lat"], cfg["lon"])})
            reference = pd.read_csv(FORECASTS / f"turbine{turbine}_february_all.csv", parse_dates=["issue_time_utc"])
            for issue in issue_times():
                result = agent.run(turbine, issue.tz_localize("UTC"))
                expected = reference.loc[reference.issue_time_utc == issue]
                np.testing.assert_allclose(result.prediction, expected.prediction, rtol=0, atol=1e-10)
                payload = json.loads(agent.last_saved_path.read_text(encoding="utf-8"))
                assert len(payload["forecast"]) == 48
                assert payload["model"]["model_version"]
                checks.append({"turbine": turbine, "issue_utc": issue.isoformat(), "rows": len(result)})
            print(f"PASS turbine {turbine}: 28 agent runs match ML reference", flush=True)
        assert len(list(Path(directory).glob("*.json"))) == 56
    print(json.dumps({"issues": len(checks), "rows": sum(row["rows"] for row in checks),
                      "matches_ml_reference": True, "tracked_forecasts_modified": False}))


if __name__ == "__main__":
    main()
