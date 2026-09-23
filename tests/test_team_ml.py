"""Integration against the committed ML artifacts and cached Previous Runs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import model
from src.config import ARTIFACTS, TURBINES, PUBLICATION_DELAY_HOURS
from src.forecast_agent import ForecastAgent
from src.ml_bridge import load_team_weather, run_team_model


ISSUE = pd.Timestamp("2026-01-31T17:00:00Z")


class TeamMLTests(unittest.TestCase):
    def agent(self, directory, turbine=1):
        cfg = TURBINES[turbine]
        return ForecastAgent(output_dir=directory, coordinates={turbine: (cfg["lat"], cfg["lon"])})

    def test_committed_models_match_ml_replay_and_preserve_metadata(self):
        for turbine in (1, 2):
            with self.subTest(turbine=turbine), tempfile.TemporaryDirectory() as directory:
                agent = self.agent(directory, turbine)
                result = agent.run(turbine, ISSUE)
                reference = pd.read_csv(ARTIFACTS.parent / "forecasts" / f"turbine{turbine}_2026-01-31.csv")
                np.testing.assert_allclose(result.prediction, reference.prediction, rtol=0, atol=1e-10)
                np.testing.assert_allclose(result.level1 + result.residual_pred, result.prediction, atol=1e-10)
                payload = json.loads(agent.last_saved_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["model"]["variant"], "direct")
                self.assertTrue(payload["model"]["model_version"].startswith(f"t{turbine}-direct"))
                self.assertNotIn("_team_archive", payload["weather"])
                self.assertEqual(len(payload["forecast"]), 48)
                self.assertEqual(payload["forecast"][0]["time"], "2026-01-31T18:00:00.000Z")
                days = np.asarray(payload["weather"]["run_day"])
                self.assertTrue((days * 24 >= np.arange(1, 49) + PUBLICATION_DELAY_HOURS).all())

    def test_future_model_rejected_and_no_result_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaisesRegex(ValueError, "labels unavailable"):
                agent.run(1, ISSUE - pd.Timedelta(days=1))
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_missing_admissible_weather_rerequests_once_then_fails_without_publishing(self):
        from src.weather import load_weather
        archive = load_weather(1).copy()
        target = ISSUE.tz_localize(None) + pd.Timedelta(hours=5)
        archive.loc[target, [c for c in archive if "previous_day" in c]] = np.nan
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.ml_bridge.load_weather", return_value=archive), \
                patch("src.ml_bridge.fetch_previous_runs", return_value=(archive.loc[[target]], {})) as live:
            agent = self.agent(directory)
            with self.assertLogs("src.forecast_agent", level="ERROR"), \
                    self.assertRaises(model.IncompleteWeatherError) as caught:
                agent.run(1, ISSUE)
            self.assertEqual(caught.exception.missing_hours, [target])
            live.assert_called_once()
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_live_rerequest_fills_gap_and_is_recorded(self):
        from src.weather import load_weather
        pinned = load_weather(1)
        holed = pinned.copy()
        target = ISSUE.tz_localize(None) + pd.Timedelta(hours=5)
        holed.loc[target, [c for c in holed if "previous_day" in c]] = np.nan
        window = pinned.loc[target - pd.Timedelta(days=1): target + pd.Timedelta(days=3)]
        meta = {"retrieved_at": "now", "sha256": "abc", "grid_latitude": 43.62, "grid_longitude": 78.48}
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.ml_bridge.load_weather", return_value=holed), \
                patch("src.ml_bridge.fetch_previous_runs", return_value=(window, meta)):
            agent = self.agent(directory)
            with self.assertLogs("src.forecast_agent", level="WARNING"):
                result = agent.run(1, ISSUE)
            reference = pd.read_csv(ARTIFACTS.parent / "forecasts" / "turbine1_2026-01-31.csv")
            np.testing.assert_allclose(result.prediction, reference.prediction, rtol=0, atol=1e-10)
            payload = json.loads(agent.last_saved_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["weather"]["live_refresh"]["sha256"], "abc")
            self.assertEqual(payload["trigger"], "retry")

    def test_update_cycle_recomputes_on_fresher_runs_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            decisions = agent.run_update_cycle(1, ISSUE, checks=(6,))
            self.assertEqual(len(decisions), 2)
            update = decisions[1]
            self.assertTrue(update["recomputed"])
            self.assertEqual(update["overlap_hours"], 42)
            self.assertIn("day2->day1", update["fresher_runs"])
            self.assertIsNotNone(update["analysis"]["revision"])
            again = agent.check_for_update(1, ISSUE + pd.Timedelta(hours=6))
            self.assertFalse(again["recomputed"])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_cli_runs_team_pipeline_and_writes_json(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-m", "src.agent_cli", "--turbine", "2", "--issue-time", "2026-02-05T17:00:00Z",
                 "--output", directory, "--updates", "6"],
                cwd=ARTIFACTS.parent, capture_output=True, text=True, check=True)
            summary = json.loads(completed.stdout)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            self.assertIn("[TOOL] analyze_forecast", completed.stderr)

    def test_fallback_to_older_admissible_run(self):
        from src.weather import load_weather
        archive = load_weather(1).copy()
        target = ISSUE.tz_localize(None) + pd.Timedelta(hours=5)
        archive.loc[target, ["wind_speed_100m_previous_day1", "temperature_2m_previous_day1"]] = np.nan
        with patch("src.ml_bridge.load_weather", return_value=archive):
            cfg = TURBINES[1]
            weather = load_team_weather(cfg["lat"], cfg["lon"], ISSUE)
        self.assertEqual(weather.loc[4, "run_day"], 2)
        result = run_team_model(1, ISSUE, weather)
        self.assertEqual(result.loc[target, "run_day"], 2)

    def test_agent_reports_fallback_to_older_run(self):
        from src.weather import load_weather
        archive = load_weather(1).copy()
        target = ISSUE.tz_localize(None) + pd.Timedelta(hours=5)
        archive.loc[target, ["wind_speed_100m_previous_day1", "temperature_2m_previous_day1"]] = np.nan
        with tempfile.TemporaryDirectory() as directory, patch("src.ml_bridge.load_weather", return_value=archive):
            agent = self.agent(directory)
            agent.run(1, ISSUE)
            flags = json.loads(agent.last_saved_path.read_text(encoding="utf-8"))["analysis"]["flags"]
        self.assertIn({"code": "older_weather_run", "hours": 1}, [{k: f[k] for k in ("code", "hours")} for f in flags])

    def test_crlf_model_load_and_custom_artifact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Reproduce Windows checkout conversion without modifying tracked artifacts.
            for name in ("model_t1.joblib", "manifest_t1.json"):
                (root / name).write_bytes((ARTIFACTS / name).read_bytes())
            text = (ARTIFACTS / "booster_t1.txt").read_text(encoding="utf-8")
            (root / "booster_t1.txt").write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            with patch("src.model.ARTIFACTS", root):
                loaded = model.load_artifacts(1)
                self.assertEqual(loaded["variant"], "direct")
            cfg = TURBINES[1]
            weather = load_team_weather(cfg["lat"], cfg["lon"], ISSUE)
            result = run_team_model(1, ISSUE, weather, root)
            self.assertEqual(len(result), 48)

    def test_coordinate_mismatch_rejected_before_fetch(self):
        with patch("src.ml_bridge.load_weather") as loader:
            with self.assertRaisesRegex(ValueError, "Coordinates"):
                load_team_weather(0., 0., ISSUE)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
