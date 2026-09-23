"""Real LightGBM/PowerCurve serialization and agent integration on synthetic data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import model
from src.features import FEATURE_COLUMNS
from src.physics import PowerCurve
from src.forecast_agent import ForecastAgent


class TeamMLTests(unittest.TestCase):
    def test_real_team_artifacts_50_weather_hours_and_agent_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            x = pd.DataFrame(rng.random((100, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
            booster = lgb.train({"objective": "regression", "verbosity": -1, "num_threads": 1,
                                 "min_data_in_leaf": 5}, lgb.Dataset(x, label=np.full(100, .02)), num_boost_round=2)
            wind = np.repeat(np.arange(0, 20, .5), 25)
            curve = PowerCurve().fit(wind, np.clip(wind / 20, 0, 1))
            with patch("src.model.ARTIFACTS", root):
                model.save_artifacts(1, curve, booster, .5)
                loaded = model.load_artifacts(1)
                self.assertAlmostEqual(loaded["blend_weight"], .5)
            issue = pd.Timestamp("2026-01-31T18:00:00Z")
            payload = {"source": "SYNTHETIC_TEST", "issued_at": issue.isoformat(),
                       "available_at": issue.isoformat(), "latitude": 0., "longitude": 0.,
                       "wind_unit": "m/s", "hourly": {
                           "time": [t.isoformat() for t in pd.date_range(issue, periods=50, freq="h")],
                           "wind_fc": [8.] * 50, "temperature": [5.] * 50}}
            archive = root / "weather.json"
            archive.write_text(json.dumps(payload), encoding="utf-8")
            history = pd.DataFrame({"power": [.4] * 30, "wind_speed": [8.] * 30},
                                   index=pd.date_range("2026-01-30T18:00:00", periods=30, freq="h"))
            agent = ForecastAgent(artifacts_dir=root, output_dir=root / "output",
                                  coordinates={1: (0., 0.)}, archive_path=archive)
            with patch("src.ml_bridge.load_history", return_value=history):
                result = agent.run(1, issue)
            self.assertEqual(len(result), 48)
            self.assertEqual(result.time.iloc[0], issue + pd.Timedelta(hours=1))
            np.testing.assert_allclose(result.residual_pred, .01)
            np.testing.assert_allclose(result.prediction, result.level1 + .01)
            saved = json.loads(agent.last_saved_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["weather"]["boundary_weather"]), 2)
            for key in payload["hourly"]:
                payload["hourly"][key] = payload["hourly"][key][1:-1]
            archive.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "50 hours"), self.assertLogs("src.forecast_agent", level="ERROR"):
                agent.run(1, issue)
            self.assertEqual(len(list((root / "output").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
