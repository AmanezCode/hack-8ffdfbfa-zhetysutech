import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.archived_weather import load_weather


class WeatherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.issue = pd.Timestamp("2026-09-23T00:00:00Z")
        self.payload = dict(source="local supplier", issued_at="2026-09-22T22:00:00Z",
                            available_at="2026-09-22T23:00:00Z", latitude=43.620384,
                            longitude=78.478910, wind_unit="m/s", hourly={
                                "time": [t.isoformat() for t in pd.date_range(
                                    self.issue + pd.Timedelta(hours=1), periods=48, freq="h")],
                                "wind_fc": [0.0] + [7.2] * 47, "temperature": [-5.0] * 48})

    def write(self, payload, name="vintage.json"):
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def load(self, payload=None, issue=None):
        return load_weather(43.620384, 78.478910, self.issue if issue is None else issue,
                            self.write(self.payload if payload is None else payload))

    def test_values_horizon_metadata(self):
        frame = self.load()
        self.assertEqual(list(frame.columns), ["time", "wind_fc", "temperature"])
        self.assertEqual(len(frame), 48)
        self.assertEqual(str(frame.time.dt.tz), "UTC")
        self.assertEqual(frame.time.iloc[0], self.issue + pd.Timedelta(hours=1))
        self.assertEqual(frame.time.iloc[-1], self.issue + pd.Timedelta(hours=48))
        self.assertEqual(frame.wind_fc.tolist(), self.payload["hourly"]["wind_fc"])
        self.assertEqual(frame.temperature.iloc[0], -5)
        self.assertEqual(frame.attrs["source"], "local supplier")
        self.assertFalse(frame.attrs["provenance_verified"])
        self.assertEqual(frame.attrs["issued_at"], pd.Timestamp(self.payload["issued_at"]))

    def test_units_offsets(self):
        self.payload["wind_unit"] = "km/h"
        for key in ("issued_at", "available_at"):
            self.payload[key] = pd.Timestamp(self.payload[key]).tz_convert("UTC+05:00").isoformat()
        self.payload["hourly"]["time"] = [pd.Timestamp(t).tz_convert("UTC+05:00").isoformat()
                                               for t in self.payload["hourly"]["time"]]
        frame = self.load(issue=self.issue.tz_convert("UTC+05:00"))
        self.assertAlmostEqual(frame.wind_fc.iloc[1], 2)
        self.assertEqual(frame.attrs["wind_unit"], "m/s")
        self.assertEqual(frame.attrs["original_wind_unit"], "km/h")

    def test_missing_provenance(self):
        for key in self.payload:
            payload = copy.deepcopy(self.payload)
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(payload)
        with self.assertRaises(ValueError):
            self.load({"hourly": {"time": [], "wind_speed_100m": []}})

    def test_bad_metadata(self):
        cases = dict(source=["", None], wind_unit=["mph", None],
                     latitude=[43.621, float("nan"), 91, True], longitude=[78.48, float("inf"), 181],
                     issued_at=["2026-09-24T00:00:00Z", "2026-09-22", None],
                     available_at=["2026-09-24T00:00:00Z", "2026-09-22", "2026-09-21T00:00:00Z"])
        for key, values in cases.items():
            for value in values:
                payload = copy.deepcopy(self.payload)
                payload[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(payload)

    def test_request_validation_and_tolerance(self):
        self.payload["latitude"] += 0.5e-6
        self.load()
        for lat, lon in [(float("nan"), 78), (91, 78), (43, 181)]:
            with self.subTest(lat=lat), self.assertRaises(ValueError):
                load_weather(lat, lon, self.issue)
        for issue in ["2026-09-23", "NaT", None, 123]:
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                load_weather(43.620384, 78.478910, issue)

    def test_bad_numbers(self):
        for key in ("wind_fc", "temperature"):
            for value in [None, float("nan"), float("inf"), -float("inf"), "7", True]:
                payload = copy.deepcopy(self.payload)
                payload["hourly"][key][3] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(payload)
        self.payload["hourly"]["wind_fc"][2] = -0.1
        with self.assertRaises(ValueError):
            self.load()

    def test_bad_shapes_and_times(self):
        for key in ("time", "wind_fc", "temperature"):
            for value in [None, [], [0] * 47, [0] * 49]:
                payload = copy.deepcopy(self.payload)
                payload["hourly"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.load(payload)
        for value in ["2026-09-23T01:00:00", "NaT", "invalid", None, self.issue.isoformat(),
                      (self.issue + pd.Timedelta(hours=2)).isoformat()]:
            payload = copy.deepcopy(self.payload)
            payload["hourly"]["time"][0] = value
            with self.subTest(time=value), self.assertRaises(ValueError):
                self.load(payload)
        self.payload["hourly"]["time"].reverse()
        with self.assertRaises(ValueError):
            self.load()

    def test_latest_eligible(self):
        self.write(self.payload, "old.json")
        latest = copy.deepcopy(self.payload)
        latest.update(issued_at=self.issue.isoformat(), available_at=self.issue.isoformat(), source="latest")
        self.write(latest, "latest.json")
        wrong = copy.deepcopy(latest)
        wrong["latitude"] = 44
        self.write(wrong, "wrong.json")
        future = copy.deepcopy(latest)
        future["available_at"] = (self.issue + pd.Timedelta(seconds=1)).isoformat()
        self.write(future, "future.json")
        self.write({"hourly": {}}, "raw.json")
        (self.root / "broken.json").write_text("{", encoding="utf-8")
        with patch("src.archived_weather.DEFAULT_ARCHIVE", self.root):
            frame = load_weather(43.620384, 78.478910, self.issue)
        self.assertEqual(frame.attrs["source"], "latest")

    def test_no_eligible_and_no_raw_fallback(self):
        self.write(self.payload, "forecast.json")
        with patch("src.archived_weather.DEFAULT_ARCHIVE", self.root / "weather"):
            with self.assertRaisesRegex(ValueError, "No eligible"):
                load_weather(43.620384, 78.478910, self.issue)
        self.payload["available_at"] = "2026-09-24T00:00:00Z"
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
