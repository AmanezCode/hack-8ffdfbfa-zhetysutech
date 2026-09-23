"""Orchestration contracts, with injected inputs and real JSON persistence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.forecast_agent import ForecastAgent, issue_timestamp


BACKEND_COLUMNS = ["time", "level1", "residual_pred", "prediction", "wind_fc", "lead_hours"]
ISSUE = pd.Timestamp("2026-01-31T23:00:00+05:00")
UTC_ISSUE = ISSUE.tz_convert("UTC")


def weather_frame(wind=6.0):
    frame = pd.DataFrame({
        "time": pd.date_range(UTC_ISSUE + pd.Timedelta(hours=1), periods=48, freq="h"),
        "wind_fc": np.full(48, wind),
        "temperature": np.full(48, -2.0),
    })
    frame.attrs["source"] = "injected verified archive"
    return frame


def components(power=0.4):
    return np.full(48, power - 0.1), np.full(48, 0.1), np.full(48, power)


def prediction_frame(weather):
    frame = weather[["time", "wind_fc"]].copy()
    frame["level1"], frame["residual_pred"], frame["prediction"] = components()
    frame["lead_hours"] = np.arange(1, 49)
    return frame[BACKEND_COLUMNS]


class ForecastAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "forecasts"
        self.weather = weather_frame()
        self.loader = Mock(side_effect=lambda *args, **kwargs: self.weather.copy(deep=True))
        self.runner = Mock(return_value=components())
        self.agent = ForecastAgent(
            output_dir=self.output, coordinates={1: (45.0, 78.0), 2: (46.0, 79.0)},
            weather_loader=self.loader, model_runner=self.runner,
        )

    def assert_no_output(self):
        self.assertIsNone(self.agent.last_saved_path)
        self.assertEqual(list(self.output.glob("*")), [])

    def test_json_roundtrip_backend_columns_and_48_utc_hours(self):
        result = self.agent.run(1, ISSUE)
        self.assertEqual(list(result.columns), BACKEND_COLUMNS)
        self.assertEqual(len(result), 48)
        self.loader.assert_called_once_with(45.0, 78.0, UTC_ISSUE)
        self.assertEqual(self.runner.call_args.args[:2], (1, UTC_ISSUE))
        path = self.agent.last_saved_path
        self.assertTrue(path.is_file())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["turbine_id"], 1)
        self.assertEqual(pd.Timestamp(payload["issue_time"]), UTC_ISSUE)
        self.assertIn(payload["run_id"], path.name)
        self.assertEqual(payload["weather"], self.weather.attrs)
        self.assertEqual(len(payload["forecast"]), 48)
        self.assertTrue(all(list(row) == BACKEND_COLUMNS for row in payload["forecast"]))
        restored = pd.DataFrame(payload["forecast"])
        restored["time"] = pd.to_datetime(restored["time"], utc=True)
        pd.testing.assert_frame_equal(restored, result, check_dtype=False)
        self.assertEqual(restored["lead_hours"].tolist(), list(range(1, 49)))
        expected = pd.date_range(UTC_ISSUE + pd.Timedelta(hours=1), periods=48, freq="h")
        self.assertTrue(pd.DatetimeIndex(restored["time"]).equals(expected))
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_issue_timestamp_normalizes_offset_and_rejects_invalid_values(self):
        self.assertEqual(issue_timestamp(ISSUE), UTC_ISSUE)
        for invalid in ("2026-01-31T23:00:00", "2026-01-31T23:01:00Z", "NaT", "not-a-date"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                issue_timestamp(invalid)

    def test_archive_path_is_forwarded(self):
        self.agent.archive_path = Path("explicit-archive.json")
        self.agent.get_archived_weather_forecast(45.0, 78.0, ISSUE)
        self.loader.assert_called_once_with(45.0, 78.0, UTC_ISSUE, archive_path=self.agent.archive_path)

    def test_bad_weather_fails_before_inference_or_saving(self):
        cases = {}
        cases["short"] = self.weather.iloc[:-1].copy()
        cases["missing_column"] = self.weather.drop(columns="temperature")
        for name in ("gap", "duplicate", "reverse", "naive", "shifted"):
            frame = self.weather.copy()
            if name == "gap":
                frame.loc[24:, "time"] += pd.Timedelta(hours=1)
            elif name == "duplicate":
                frame.loc[1, "time"] = frame.loc[0, "time"]
            elif name == "reverse":
                frame = frame.iloc[::-1]
            elif name == "naive":
                frame["time"] = frame["time"].dt.tz_localize(None)
            else:
                frame["time"] -= pd.Timedelta(hours=1)
            cases[name] = frame
        for column in ("wind_fc", "temperature"):
            for value in (np.nan, np.inf, -np.inf):
                frame = self.weather.copy()
                frame.loc[12, column] = value
                cases[f"{column}_{value}"] = frame
        frame = self.weather.copy()
        frame.loc[0, "wind_fc"] = -1.0
        cases["negative_wind"] = frame
        for name, frame in cases.items():
            with self.subTest(case=name):
                self.weather = frame
                self.loader.reset_mock()
                with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(ValueError):
                    self.agent.run(1, ISSUE, max_attempts=3)
                self.assertEqual(self.loader.call_count, 1)
                self.runner.assert_not_called()
                self.assert_no_output()

    def test_prediction_validation_rejects_invalid_components(self):
        for column in BACKEND_COLUMNS[1:]:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(column=column, value=value):
                    pred = prediction_frame(self.weather)
                    pred[column] = pred[column].astype(float)
                    pred.loc[10, column] = value
                    with self.assertRaises(ValueError):
                        self.agent.validate_prediction(pred, self.weather)

    def test_prediction_range_wind_composition_and_schema_failures(self):
        cases = [
            ("prediction", -0.01), ("prediction", 1.01),
            ("prediction", 0.6), ("wind_fc", 7.0), ("lead_hours", 0),
            ("time", self.weather.loc[0, "time"] + pd.Timedelta(hours=1)),
        ]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                pred = prediction_frame(self.weather)
                pred.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.agent.validate_prediction(pred, self.weather)
        for pred in (prediction_frame(self.weather).iloc[:-1],
                     prediction_frame(self.weather).drop(columns="residual_pred")):
            with self.assertRaises(ValueError):
                self.agent.validate_prediction(pred, self.weather)

    def test_clipped_composition_is_valid(self):
        pred = prediction_frame(self.weather)
        pred.loc[0, ["level1", "residual_pred", "prediction"]] = [0.9, 0.2, 1.0]
        pred.loc[1, ["level1", "residual_pred", "prediction"]] = [0.1, -0.2, 0.0]
        self.agent.validate_prediction(pred, self.weather)

    def test_zero_wind_power_threshold(self):
        weather = weather_frame(wind=0.0)
        for power in (0.0, 0.05, 0.051, 0.4):
            with self.subTest(power=power):
                pred = prediction_frame(weather)
                pred["level1"] = power
                pred["residual_pred"] = 0.0
                pred["prediction"] = power
                if power <= 0.05:
                    self.agent.validate_prediction(pred, weather)
                else:
                    with self.assertRaisesRegex(ValueError, "zero"):
                        self.agent.validate_prediction(pred, weather)

    def test_missing_coordinates_and_unsupported_turbine_fail_fast(self):
        for coordinates, turbine in (({}, 1), ({1: (45.0, 78.0)}, 2), ({3: (45.0, 78.0)}, 3)):
            with self.subTest(coordinates=coordinates, turbine=turbine):
                self.agent.coordinates = coordinates
                with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(ValueError):
                    self.agent.run(turbine, ISSUE)
                self.loader.assert_not_called()
                self.runner.assert_not_called()
                self.assert_no_output()

    def test_transient_weather_or_model_errors_retry_and_reload(self):
        for stage in ("weather", "model"):
            for error in (TimeoutError, ConnectionError):
                with self.subTest(stage=stage, error=error):
                    self.loader.reset_mock()
                    self.runner.reset_mock()
                    self.loader.side_effect = [error("temporary"), self.weather.copy()] if stage == "weather" else lambda *a: self.weather.copy()
                    self.runner.side_effect = [error("temporary"), components()] if stage == "model" else None
                    with self.assertLogs("src.forecast_agent", level="WARNING"):
                        result = self.agent.run(1, ISSUE, max_attempts=2)
                    self.assertEqual(len(result), 48)
                    self.assertEqual(self.loader.call_count, 2)
                    self.assertEqual(self.runner.call_count, 2 if stage == "model" else 1)

    def test_transient_errors_exhaust_attempts_without_saving(self):
        self.runner.side_effect = TimeoutError("still unavailable")
        with self.assertLogs("src.forecast_agent", level="WARNING"), self.assertRaises(TimeoutError):
            self.agent.run(1, ISSUE, max_attempts=3)
        self.assertEqual(self.loader.call_count, 3)
        self.assertEqual(self.runner.call_count, 3)
        self.assert_no_output()

    def test_validation_and_missing_inputs_are_not_retried(self):
        for stage in (self.loader, self.runner):
            for error in (ValueError, FileNotFoundError):
                with self.subTest(stage="weather" if stage is self.loader else "model", error=error):
                    self.loader.reset_mock()
                    self.runner.reset_mock()
                    self.loader.side_effect = lambda *a: self.weather.copy()
                    self.runner.side_effect = None
                    stage.side_effect = error("invalid or missing input")
                    with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(error):
                        self.agent.run(1, ISSUE, max_attempts=3)
                    self.assertEqual(self.loader.call_count, 1)
                    self.assertEqual(self.runner.call_count, 0 if stage is self.loader else 1)
                    self.assert_no_output()

    def test_invalid_model_output_is_not_retried_or_saved(self):
        self.runner.return_value = (np.full(48, 0.3), np.full(48, 0.1), np.full(48, 0.7))
        with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(ValueError):
            self.agent.run(1, ISSUE, max_attempts=3)
        self.loader.assert_called_once()
        self.runner.assert_called_once()
        self.assert_no_output()

    def test_attempt_bounds(self):
        for attempts in (0, -1, 6):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                self.agent.run(1, ISSUE, max_attempts=attempts)
        self.loader.assert_not_called()

    def test_model_outputs_must_be_48_element_vectors(self):
        for invalid in (0.4, np.full(47, 0.4), np.full((48, 1), 0.4)):
            with self.subTest(shape=np.shape(invalid)):
                self.loader.reset_mock()
                self.runner.reset_mock()
                self.runner.return_value = (invalid, np.full(48, 0.1), np.full(48, 0.4))
                with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(ValueError):
                    self.agent.run(1, ISSUE, max_attempts=3)
                self.loader.assert_called_once()
                self.runner.assert_called_once()
                self.assert_no_output()

    def test_repeated_run_versions_and_reloads_changed_inputs(self):
        first = self.agent.run(2, ISSUE)
        first_path = self.agent.last_saved_path
        first_bytes = first_path.read_bytes()
        self.weather["wind_fc"] = 9.0
        self.runner.return_value = components(0.6)
        second = self.agent.run(2, ISSUE)
        second_path = self.agent.last_saved_path
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(len(list(self.output.glob("*.json"))), 2)
        self.assertEqual(self.loader.call_count, 2)
        self.assertEqual(self.runner.call_count, 2)
        self.assertTrue((first["wind_fc"] == 6.0).all())
        self.assertTrue((second["wind_fc"] == 9.0).all())
        self.assertTrue((second["prediction"] == 0.6).all())
        payload = json.loads(second_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["turbine_id"], 2)
        self.assertEqual(payload["forecast"][0]["wind_fc"], 9.0)
        self.assertNotEqual(payload["run_id"], json.loads(first_bytes)["run_id"])

    def test_invalid_timestamp_after_success_clears_last_saved_path(self):
        self.agent.run(1, ISSUE)
        saved_path = self.agent.last_saved_path
        self.assertIsNotNone(saved_path)
        saved_bytes = saved_path.read_bytes()
        self.loader.reset_mock()
        self.runner.reset_mock()

        with self.assertLogs("src.forecast_agent", level="ERROR"), self.assertRaises(ValueError):
            self.agent.run(1, "not-a-date")

        self.assertIsNone(self.agent.last_saved_path)
        self.loader.assert_not_called()
        self.runner.assert_not_called()
        self.assertEqual(list(self.output.glob("*")), [saved_path])
        self.assertEqual(saved_path.read_bytes(), saved_bytes)

    def test_model_receives_copy_of_weather(self):
        def mutate_weather(turbine, stamp, weather):
            weather["wind_fc"] = 99.0
            return components()
        self.runner.side_effect = mutate_weather
        result = self.agent.forecast(1, ISSUE)
        self.assertTrue((result["wind_fc"] == 6.0).all())
        self.assertTrue((self.weather["wind_fc"] == 6.0).all())

    def test_direct_model_tool_uses_injected_inputs_without_saving(self):
        actual = self.agent.run_forecast_model(1, ISSUE)
        for array, expected in zip(actual, components()):
            np.testing.assert_array_equal(array, expected)
        self.loader.assert_called_once()
        self.runner.assert_called_once()
        self.assert_no_output()

    def test_direct_save_rejects_wrong_issue_and_invalid_prediction(self):
        pred = prediction_frame(self.weather)
        with self.assertRaises(ValueError):
            self.agent.save_forecast(1, ISSUE + pd.Timedelta(hours=1), pred)
        pred.loc[0, "prediction"] = 2.0
        with self.assertRaises(ValueError):
            self.agent.save_forecast(1, ISSUE, pred)
        self.assert_no_output()


if __name__ == "__main__":
    unittest.main()
