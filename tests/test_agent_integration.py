"""Exercise the CLI with real file adapters and a synthetic serialized estimator."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.forecast_agent import ROOT, COLUMNS


class AgentIntegrationTests(unittest.TestCase):
    def test_cli_loads_files_predicts_and_saves_backend_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = LinearRegression().fit(pd.DataFrame({"wind_fc": [0., 5., 10.]}), [0., .25, .5])
            joblib.dump(model, root / "model_t1.joblib")
            (root / "residual_t1.txt").write_text("0.01", encoding="utf-8")
            (root / "metadata_t1.json").write_text(json.dumps({"residual": {"format": "scalar"}}), encoding="utf-8")
            issue = pd.Timestamp("2026-01-31T18:00:00Z")
            payload = {"source": "SYNTHETIC_TEST_ONLY", "issued_at": issue.isoformat(),
                       "available_at": issue.isoformat(), "latitude": 0., "longitude": 0.,
                       "wind_unit": "m/s", "hourly": {
                           "time": [t.isoformat() for t in pd.date_range(issue + pd.Timedelta(hours=1), periods=48, freq="h")],
                           "wind_fc": [5.] * 48, "temperature": [10.] * 48}}
            weather_path = root / "weather.json"
            weather_path.write_text(json.dumps(payload), encoding="utf-8")
            args = [sys.executable, "-m", "src.agent_cli", "--turbine", "1", "--lat", "0", "--lon", "0",
                    "--issue-time", "2026-01-31T23:00:00+05:00", "--weather", str(weather_path),
                    "--artifacts", str(root), "--model-backend", "generic", "--output", str(root / "out")]
            process = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            output = json.loads(process.stdout)
            saved = json.loads(Path(output["path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(saved["forecast"]), 48)
            self.assertEqual(list(saved["forecast"][0]), COLUMNS)
            self.assertAlmostEqual(saved["forecast"][0]["prediction"], .26)
            self.assertEqual(saved["forecast"][-1]["lead_hours"], 48)
            self.assertEqual(saved["weather"]["source"], "SYNTHETIC_TEST_ONLY")
            payload["available_at"] = (issue + pd.Timedelta(hours=1)).isoformat()
            weather_path.write_text(json.dumps(payload), encoding="utf-8")
            failed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(len(list((root / "out").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
