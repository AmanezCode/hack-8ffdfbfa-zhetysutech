"""Temporary serialized fixtures only; no production artifacts or training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.artifact_adapter import load_artifacts, run_forecast_model


class Estimator:
    """Serializable deterministic estimator proving ordered predict invocation."""
    def __init__(self, names=None, output=None):
        if names is not None:
            self.feature_names_in_ = np.array(names)
        self.output = output

    def predict(self, frame):
        if self.output is not None:
            return self.output
        return frame.iloc[:, 0].to_numpy() - frame.iloc[:, 1].to_numpy()


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.weather = pd.DataFrame({"b": [0.2, 0.3], "a": [0.5, 1.5]})

    def pair(self, model=None, metadata=None, residual="0.1", turbine=1):
        joblib.dump(model if model is not None else Estimator(["a", "b"]),
                    self.directory / f"model_t{turbine}.joblib")
        (self.directory / f"residual_t{turbine}.txt").write_text(residual, encoding="utf-8")
        (self.directory / f"metadata_t{turbine}.json").write_text(json.dumps(
            metadata if metadata is not None else {"residual": {"format": "scalar"}}), encoding="utf-8")

    def forecast(self, weather=None):
        return run_forecast_model(1, "2026-01-01", self.weather if weather is None else weather,
                                  load_artifacts(self.directory))

    def test_serialized_estimator_order_and_clipping(self):
        self.pair()
        level1, residual, prediction = self.forecast()
        np.testing.assert_allclose(level1, [0.3, 1.2])
        np.testing.assert_allclose(residual, [0.1, 0.1])
        np.testing.assert_allclose(prediction, [0.4, 1])
        self.pair(residual="-2")
        np.testing.assert_array_equal(self.forecast()[2], [0, 0])

    def test_explicit_order_without_estimator_names(self):
        self.pair(Estimator(), {"feature_names": ["b", "a"], "residual": {"format": "scalar"}})
        np.testing.assert_allclose(self.forecast()[0], [-0.3, -1.2])

    def test_scalar_does_not_require_lightgbm_and_overflow_is_rejected(self):
        self.pair()
        with patch.dict(sys.modules, {"lightgbm": None}):
            np.testing.assert_allclose(self.forecast()[2], [0.4, 1])
        self.pair(Estimator(["a", "b"], [1e308, 1e308]), residual="1e308")
        with self.assertRaisesRegex(ValueError, "combined prediction"):
            self.forecast()

    def test_utc_time_features(self):
        weather = pd.DataFrame({"time": ["2026-01-02T01:00:00+05:00"], "hour": [999]})
        original = weather.copy(deep=True)
        for name, expected in [("hour", 20), ("day_of_year", 1), ("lead_hours", 20)]:
            with self.subTest(name=name):
                self.pair(Estimator([name, "zero"]))
                np.testing.assert_allclose(self.forecast(weather.assign(zero=0))[0], [expected])
        pd.testing.assert_frame_equal(weather, original)
        self.pair(Estimator(["lead_hours", "zero"]))
        result = run_forecast_model(1, "2026-01-02T00:00:00+05:00", weather.assign(zero=0), load_artifacts(self.directory))
        np.testing.assert_allclose(result[0], [1])

    def test_missing_pairs(self):
        self.assertEqual(load_artifacts(self.directory / "absent"), {})
        joblib.dump(Estimator(), self.directory / "model_t1.joblib")
        self.pair(turbine=7)
        artifacts = load_artifacts(self.directory)
        self.assertEqual(set(artifacts), {7})
        with self.assertRaisesRegex(FileNotFoundError, "turbine 1"):
            run_forecast_model(1, "2026-01-01", self.weather, artifacts)

    def test_missing_metadata_never_guesses_scalar(self):
        self.pair()
        (self.directory / "metadata_t1.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "format must be declared"):
            load_artifacts(self.directory)

    def test_invalid_artifacts(self):
        cases = [({}, "0"), ({"residual": {}}, "0"),
                 ({"residual": {"format": "unknown"}}, "0"),
                 ({"residual": {"format": "scalar"}}, "nan"),
                 ({"residual": {"format": "scalar"}}, "1 2"),
                 ({"feature_names": ["b", "a"], "residual": {"format": "scalar"}}, "0"),
                 ({"feature_names": ["a", "a"], "residual": {"format": "scalar"}}, "0")]
        for metadata, residual in cases:
            with self.subTest(metadata=metadata, residual=residual):
                self.pair(metadata=metadata, residual=residual)
                with self.assertRaises(ValueError):
                    load_artifacts(self.directory)
        self.pair(Estimator())
        with self.assertRaisesRegex(ValueError, "order cannot be guessed"):
            load_artifacts(self.directory)
        self.pair()
        (self.directory / "model_t1.joblib").write_bytes(b"invalid serialization")
        with self.assertRaisesRegex(ValueError, "turbine 1"):
            load_artifacts(self.directory)
        self.pair()
        (self.directory / "metadata_t1.json").write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_artifacts(self.directory)

    def test_invalid_weather_and_outputs(self):
        self.pair()
        for weather in [self.weather.drop(columns="a"), self.weather.assign(a=np.nan),
                        self.weather.assign(a=np.inf), self.weather.assign(a="bad"),
                        self.weather.iloc[:0], pd.DataFrame([[1, 2]], columns=["a", "a"])]:
            with self.subTest(weather=weather):
                with self.assertRaises(ValueError):
                    self.forecast(weather)
        for output in [1, [1], [[1], [2]], [np.nan, 1], [np.inf, 1]]:
            with self.subTest(output=output):
                self.pair(Estimator(["a", "b"], output))
                with self.assertRaisesRegex(ValueError, "level1"):
                    self.forecast()
        self.pair()
        with self.assertRaisesRegex(ValueError, "issue_time"):
            run_forecast_model(1, "bad", self.weather, load_artifacts(self.directory))
        self.pair(Estimator(["hour", "a"]))
        for weather in [self.weather, self.weather.assign(time="bad"), self.weather.assign(time=pd.NaT)]:
            with self.assertRaises(ValueError):
                self.forecast(weather)

    def test_lightgbm_optional_import_and_prediction_contract(self):
        metadata = {"residual": {"format": "lightgbm", "feature_names": ["b", "a"]}}
        self.pair(metadata=metadata, residual="test-only booster placeholder")
        with patch.dict(sys.modules, {"lightgbm": None}):
            with self.assertRaisesRegex(ImportError, "optional package"):
                load_artifacts(self.directory)
        # Mock only the optional library boundary; joblib estimator remains real.
        booster = SimpleNamespace(feature_name=lambda: ["b", "a"],
                                  predict=lambda frame: frame.iloc[:, 0].to_numpy())
        module = SimpleNamespace(Booster=lambda model_file: booster)
        with patch.dict(sys.modules, {"lightgbm": module}):
            np.testing.assert_allclose(self.forecast()[1], [0.2, 0.3])
            for invalid in [np.array([[1], [2]]), np.array([np.nan, 0])]:
                booster.predict = lambda frame: invalid
                with self.assertRaisesRegex(ValueError, "residual_pred"):
                    self.forecast()
            booster.feature_name = lambda: ["a", "b"]
            with self.assertRaisesRegex(ValueError, "LightGBM feature order"):
                load_artifacts(self.directory)


if __name__ == "__main__":
    unittest.main()
