"""ForecastAgent: the full forecasting cycle required by the track.

get weather -> validate -> run model -> validate -> analyse -> publish, and
re-check the inputs later: when fresher weather becomes admissible for hours
already forecast, the agent recomputes and publishes a new version with a
revision report; when nothing changed it keeps the published forecast.

The numbers always come from the ML pipeline; the agent decides when to run
it, checks and explains the result, and reacts to missing inputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout as RequestsTimeout

from .config import SCADA_UTC_OFFSET_HOURS, UPDATE_INTERVAL_HOURS
from .model import IncompleteWeatherError
from .weather import run_day_for_lead

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["time", "level1", "residual_pred", "prediction", "wind_fc", "lead_hours"]
HORIZON = 48
LOG = logging.getLogger(__name__)

# Ramp: a change of at least 30 % of rated power within 3 h, a common operator definition.
RAMP_THRESHOLD = 0.3
RAMP_WINDOW_HOURS = 3
ZERO_WIND_POWER = 0.05
TRANSIENT_ERRORS = (TimeoutError, ConnectionError, RequestsTimeout, RequestsConnectionError)


def issue_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp != stamp.floor("h"):
        raise ValueError("issue_time must be a timezone-aware whole hour")
    return stamp.tz_convert("UTC")


def _scada(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def _utc(values) -> pd.Series:
    """Parsed JSON times and in-memory times can differ in resolution; merge keys must match."""
    return pd.to_datetime(pd.Series(values), utc=True).astype("datetime64[ns, UTC]")


class ForecastAgent:
    """Tools: get_archived_weather_forecast, validate_weather, run_forecast_model,
    validate_prediction, analyze_forecast, save_forecast. Orchestration: run,
    check_for_update, run_update_cycle."""

    def __init__(self, output_dir: str | Path = ROOT / "forecasts" / "agent",
                 artifacts_dir: str | Path = ROOT / "artifacts", *,
                 coordinates: Mapping[int, tuple[float, float]] | None = None,
                 weather_loader: Callable | None = None,
                 model_runner: Callable | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.coordinates = dict(coordinates or {})
        self.weather_loader = weather_loader
        self.model_runner = model_runner
        self.last_saved_path: Path | None = None
        self._model_metadata: dict = {}

    # ---------------------------------------------------------------- tools

    def get_archived_weather_forecast(self, lat: float, lon: float, issue_date: str | pd.Timestamp, *,
                                      refresh: bool = False) -> pd.DataFrame:
        stamp = issue_timestamp(issue_date)
        loader = self.weather_loader
        if loader is None:
            from .ml_bridge import load_team_weather
            loader = load_team_weather
        weather = loader(lat, lon, stamp, refresh=True) if refresh else loader(lat, lon, stamp)
        self.validate_weather(weather)
        expected = pd.date_range(stamp + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
        if not pd.DatetimeIndex(weather["time"]).equals(expected):
            raise ValueError("Weather must cover issue_time + 1..48 hours in UTC")
        return weather

    @staticmethod
    def validate_weather(weather: pd.DataFrame) -> None:
        required = {"time", "wind_fc", "temperature"}
        missing = required - set(weather.columns)
        if missing:
            raise ValueError(f"Weather is missing columns: {sorted(missing)}")
        if len(weather) != HORIZON:
            raise ValueError(f"Expected 48 hourly weather rows, got {len(weather)}")
        if weather[sorted(required)].isna().any().any():
            raise ValueError("Weather contains NaN values")
        if not weather["time"].is_monotonic_increasing or weather["time"].duplicated().any():
            raise ValueError("Weather timestamps must be unique and increasing")
        times = pd.DatetimeIndex(weather["time"])
        if times.tz is None or not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
            raise ValueError("Weather requires timezone-aware consecutive hourly timestamps")
        numeric = weather.select_dtypes(include="number")
        if weather.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Weather contains missing or nonfinite values")
        if (weather["wind_fc"] < 0).any():
            raise ValueError("Weather wind must be nonnegative")

    def run_forecast_model(self, turbine_id: int, issue_time: str | pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stamp = issue_timestamp(issue_time)
        return self._predict(turbine_id, stamp, self._weather(turbine_id, stamp))

    @staticmethod
    def validate_prediction(pred: pd.DataFrame, weather: pd.DataFrame) -> None:
        """Hard checks only: a forecast that fails them is never published."""
        ForecastAgent.validate_weather(weather)
        missing = set(COLUMNS) - set(pred.columns)
        if missing:
            raise ValueError(f"Prediction is missing columns: {sorted(missing)}")
        if len(pred) != len(weather):
            raise ValueError("Prediction and weather lengths differ")
        if not pd.DatetimeIndex(pred["time"]).equals(pd.DatetimeIndex(weather["time"])):
            raise ValueError("Prediction timestamps differ from weather")
        if not np.isfinite(pred[COLUMNS[1:]].to_numpy(dtype=float)).all():
            raise ValueError("Prediction components must be finite")
        if not np.array_equal(pred["lead_hours"], np.arange(1, HORIZON + 1)):
            raise ValueError("lead_hours must be 1..48")
        if not np.allclose(pred["wind_fc"], weather["wind_fc"], rtol=0, atol=1e-8):
            raise ValueError("Prediction wind differs from weather")
        values = pred["prediction"].to_numpy(dtype=float)
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Prediction must be in [0, 1]")
        expected = np.clip(pred["level1"] + pred["residual_pred"], 0, 1)
        if not np.allclose(values, expected, rtol=0, atol=1e-8):
            raise ValueError("prediction does not match level1 + residual_pred")

    def analyze_forecast(self, pred: pd.DataFrame, weather: pd.DataFrame | None = None,
                         previous: dict | None = None) -> dict:
        """Operator-facing reading of a validated forecast (normalised power = share of rated capacity)."""
        card = pred.attrs.get("model", {})
        values = pred["prediction"].to_numpy(dtype=float)
        lead = pred["lead_hours"].to_numpy()
        scada = _scada(pred["time"])
        expected_error = card.get("expected_abs_error_by_lead")
        band = np.asarray(expected_error, dtype=float)[lead - 1] if expected_error else None

        days = []
        for day in sorted(set(scada.date)):
            mask = scada.date == day
            peak = int(np.argmax(np.where(mask, values, -1)))
            days.append({
                "date_scada": str(day),
                "hours": int(mask.sum()),
                "capacity_factor": round(float(values[mask].mean()), 4),
                "full_load_hours": round(float(values[mask].sum()), 2),
                "peak_scada": scada[peak].strftime("%Y-%m-%d %H:%M"),
                "peak_power": round(float(values[peak]), 4),
                "expected_mae": round(float(band[mask].mean()), 4) if band is not None else None,
            })

        ramps = []
        change = values[RAMP_WINDOW_HOURS:] - values[:-RAMP_WINDOW_HOURS]
        for start in np.flatnonzero(np.abs(change) >= RAMP_THRESHOLD):
            if ramps and start <= ramps[-1]["_end"]:  # overlapping windows are one event
                ramps[-1]["_end"] = start + RAMP_WINDOW_HOURS
                if abs(change[start]) > abs(ramps[-1]["change"]):
                    ramps[-1]["change"] = round(float(change[start]), 3)
                continue
            ramps.append({"start_scada": scada[start].strftime("%Y-%m-%d %H:%M"),
                          "change": round(float(change[start]), 3), "_end": start + RAMP_WINDOW_HOURS})
        for ramp in ramps:
            ramp["direction"] = "up" if ramp["change"] > 0 else "down"
            ramp.pop("_end")

        flags = []
        wind = pred["wind_fc"].to_numpy(dtype=float)
        calm = int(((wind <= 0) & (values > ZERO_WIND_POWER)).sum())
        if calm:
            flags.append({"code": "zero_wind_power", "hours": calm,
                          "text": f"{calm} h with zero forecast wind but power > {ZERO_WIND_POWER:.0%} (hour average around a calm)"})
        deviation_limit = card.get("deviation_q99")
        if deviation_limit:
            unusual = int((np.abs(values - pred["level1"].to_numpy(dtype=float)) > deviation_limit).sum())
            if unusual:
                flags.append({"code": "curve_deviation", "hours": unusual, "limit": round(float(deviation_limit), 4),
                              "text": f"{unusual} h deviate from the power curve more than the CV 99th percentile ({deviation_limit:.2f})"})
        if weather is not None and "run_day" in weather:
            older = int((weather["run_day"].to_numpy() > run_day_for_lead(lead)).sum())
            if older:
                flags.append({"code": "older_weather_run", "hours": older,
                              "text": f"{older} h use an older weather run than usual (freshest run missing in the archive)"})

        revision = None
        if previous is not None:
            old = pd.DataFrame(previous["forecast"])
            old["time"] = _utc(old["time"])
            joined = pd.DataFrame({"time": _utc(pred["time"]).to_numpy(), "new": values}).merge(old[["time", "prediction"]], on="time")
            if len(joined):
                delta = joined["new"] - joined["prediction"]
                worst = int(np.argmax(np.abs(delta.to_numpy())))
                revision = {
                    "previous_run_id": previous["run_id"],
                    "previous_issue_time": previous["issue_time"],
                    "overlap_hours": int(len(joined)),
                    "mean_abs_change": round(float(delta.abs().mean()), 4),
                    "max_abs_change": round(float(abs(delta.iloc[worst])), 4),
                    "max_change_scada": _scada(joined["time"])[worst].strftime("%Y-%m-%d %H:%M"),
                    "energy_change_full_load_hours": round(float(delta.sum()), 2),
                }

        first = days[0]
        span = "" if first["hours"] == 24 else f" ({first['hours']} h left)"
        summary = (f"{first['date_scada']}{span}: capacity factor {first['capacity_factor']:.0%}, "
                   f"{first['full_load_hours']:.1f} full-load hours, peak {first['peak_power']:.0%} at {first['peak_scada'][-5:]}")
        if first["expected_mae"] is not None:
            summary += f", expected error +/-{first['expected_mae']:.0%}"
        if ramps:
            summary += f"; {len(ramps)} ramp(s), largest {max(abs(r['change']) for r in ramps):.0%}"
        if revision:
            summary += f"; revised {revision['mean_abs_change']:.1%} on average vs previous forecast"

        return {"summary": summary, "days": days, "ramps": ramps, "flags": flags, "revision": revision,
                "expected_abs_error": band.round(4).tolist() if band is not None else None,
                "cv_accuracy": card.get("cv_accuracy")}

    def save_forecast(self, turbine_id: int, issue_time: str | pd.Timestamp, pred: pd.DataFrame) -> Path:
        stamp = issue_timestamp(issue_time)
        if turbine_id not in (1, 2):
            raise ValueError("Unsupported turbine_id")
        # Validate even when called as an independent tool.
        weather = pred[["time", "wind_fc"]].copy()
        weather["temperature"] = 0.0
        self.validate_prediction(pred, weather)
        expected = pd.date_range(stamp + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
        if not pd.DatetimeIndex(pred["time"]).equals(expected):
            raise ValueError("Prediction timestamps differ from issue_time")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        path = self.output_dir / f"turbine_{turbine_id}_{stamp.strftime('%Y%m%dT%H%M%SZ')}_{run_id}.json"
        records = json.loads(pred[COLUMNS].to_json(orient="records", date_format="iso"))
        extra = pred.attrs.get("agent", {})
        payload = {"turbine_id": turbine_id, "issue_time": stamp.isoformat(), "run_id": run_id,
                   "weather": pred.attrs.get("weather", {}), "model": pred.attrs.get("model", {}),
                   "forecast": records, **extra}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
        self.last_saved_path = path
        LOG.info("[TOOL] save_forecast: turbine %s, 48 h, version %s -> %s", turbine_id, extra.get("version", 1), path.name)
        return path

    # -------------------------------------------------------- orchestration

    def forecast(self, turbine_id: int, issue_time: str | pd.Timestamp, *, refresh_weather: bool = False,
                 trigger: str = "scheduled") -> pd.DataFrame:
        """One pass of the cycle for one issue time."""
        self.last_saved_path = None
        stamp = issue_timestamp(issue_time)
        weather = self._weather(turbine_id, stamp, refresh=refresh_weather)
        return self._forecast_from(turbine_id, stamp, weather, trigger)

    def run(self, turbine_id: int, issue_time: str | pd.Timestamp, *, max_attempts: int = 2,
            refresh_weather: bool = False) -> pd.DataFrame:
        """Run the cycle with bounded retries.

        Transient I/O errors are retried. Missing admissible weather triggers one
        live re-request to Open-Meteo; if hours are still missing the error is
        raised and nothing is published. Invalid inputs or outputs are never
        repaired by inventing data.
        """
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        refresh = refresh_weather
        for attempt in range(max_attempts):
            try:
                return self.forecast(turbine_id, issue_time, refresh_weather=refresh,
                                     trigger="scheduled" if attempt == 0 else "retry")
            except IncompleteWeatherError as error:
                if refresh or attempt + 1 == max_attempts:
                    LOG.error("[AGENT] weather still missing for %d h after re-request, not publishing", len(error.missing_hours))
                    raise
                LOG.warning("[AGENT] admissible weather missing for %d h -> re-requesting Open-Meteo live", len(error.missing_hours))
                refresh = True
            except (FileNotFoundError, ValueError):
                self.last_saved_path = None
                LOG.exception("[AGENT] forecast failed for turbine %s", turbine_id)
                raise
            except TRANSIENT_ERRORS:
                LOG.warning("[AGENT] transient failure, attempt %s of %s", attempt + 1, max_attempts)
                if attempt + 1 == max_attempts:
                    raise
        raise RuntimeError("Unreachable")

    def check_for_update(self, turbine_id: int, check_time: str | pd.Timestamp, *, refresh_weather: bool = False) -> dict:
        """Recompute only if the inputs admissible now differ from those of the published forecast."""
        check = issue_timestamp(check_time)
        previous = self.latest_published(turbine_id, check)
        weather = self._weather(turbine_id, check, refresh=refresh_weather)
        decision = {"check_time": check.isoformat(), "previous_run_id": previous["run_id"] if previous else None,
                    "overlap_hours": 0, "changed_hours": 0, "fresher_runs": {}, "model_changed": False, "recomputed": False}

        if previous is None:
            LOG.info("[AGENT] no published forecast covers %s -> forecasting", check)
            result = self._forecast_from(turbine_id, check, weather, "initial")
            return decision | {"recomputed": True, "run_id": self._run_id(), "analysis": result.attrs["agent"]["analysis"]}

        old = pd.DataFrame(previous.get("inputs", []))
        if old.empty:
            changed = pd.DatetimeIndex([])
        else:
            old["time"] = _utc(old["time"])
            current = weather.assign(time=_utc(weather["time"]).to_numpy())
            joined = current.merge(old, on="time", suffixes=("", "_old"))
            decision["overlap_hours"] = int(len(joined))
            differs = ~np.isclose(joined["wind_fc"], joined["wind_fc_old"], atol=1e-9) | \
                      ~np.isclose(joined["temperature"], joined["temperature_old"], atol=1e-9)
            if "run_day" in joined and "run_day_old" in joined:
                differs |= joined["run_day"].to_numpy() != joined["run_day_old"].to_numpy()
                moves = joined.loc[differs, ["run_day_old", "run_day"]].astype(int)
                decision["fresher_runs"] = {f"day{a}->day{b}": int(n) for (a, b), n in moves.value_counts().items()}
            changed = pd.DatetimeIndex(joined.loc[differs, "time"])
        decision["changed_hours"] = int(len(changed))

        current_version = self._current_model_version(turbine_id)
        decision["model_changed"] = bool(current_version and current_version != previous.get("model", {}).get("model_version"))

        if not len(changed) and not decision["model_changed"]:
            LOG.info("[AGENT] inputs for %d overlapping hours unchanged -> keeping published forecast %s",
                     decision["overlap_hours"], previous["run_id"][:8])
            return decision

        reason = "model updated" if decision["model_changed"] else f"fresher weather for {len(changed)} of {decision['overlap_hours']} hours {decision['fresher_runs']}"
        LOG.info("[AGENT] %s -> recomputing", reason)
        result = self._forecast_from(turbine_id, check, weather, "input_update")
        return decision | {"recomputed": True, "run_id": self._run_id(), "analysis": result.attrs["agent"]["analysis"]}

    def run_update_cycle(self, turbine_id: int, issue_time: str | pd.Timestamp, *,
                         checks: tuple[int, ...] = (UPDATE_INTERVAL_HOURS, 2 * UPDATE_INTERVAL_HOURS, 3 * UPDATE_INTERVAL_HOURS),
                         refresh_weather: bool = False) -> list[dict]:
        """Scheduled forecast, then re-checks as time passes (replayed on the historical clock)."""
        stamp = issue_timestamp(issue_time)
        first = self.run(turbine_id, stamp, refresh_weather=refresh_weather)
        decisions = [{"check_time": stamp.isoformat(), "recomputed": True, "trigger": "scheduled",
                      "run_id": self._run_id(), "analysis": first.attrs["agent"]["analysis"]}]
        for hours in checks:
            decisions.append(self.check_for_update(turbine_id, stamp + pd.Timedelta(hours=hours), refresh_weather=refresh_weather))
        return decisions

    def latest_published(self, turbine_id: int, at: pd.Timestamp) -> dict | None:
        """Most recent forecast issued at or before `at` (latest version if re-run)."""
        if not self.output_dir.is_dir():
            return None
        best = None
        for path in self.output_dir.glob(f"turbine_{turbine_id}_*.json"):
            try:
                issued = pd.Timestamp(path.name.split("_")[2]).tz_convert("UTC")
            except (IndexError, ValueError):
                continue
            if issued <= at and (best is None or (issued, path.stat().st_mtime_ns) > best[0]):
                best = ((issued, path.stat().st_mtime_ns), path)
        return json.loads(best[1].read_text(encoding="utf-8")) if best else None

    # ------------------------------------------------------------ internals

    def _weather(self, turbine_id: int, issue_time: pd.Timestamp, *, refresh: bool = False) -> pd.DataFrame:
        if turbine_id not in (1, 2) or turbine_id not in self.coordinates:
            raise ValueError("Provide verified coordinates for turbine 1 or 2")
        lat, lon = self.coordinates[turbine_id]
        LOG.info("[TOOL] get_archived_weather_forecast: turbine %s, issue %s%s", turbine_id,
                 issue_time.strftime("%Y-%m-%d %H:%M UTC"), " (live re-request)" if refresh else "")
        weather = self.get_archived_weather_forecast(lat, lon, issue_time, refresh=refresh)
        LOG.info("[TOOL] validate_weather: 48 h ok%s", f", runs {sorted(set(int(d) for d in weather['run_day']))} day(s) old"
                 if "run_day" in weather else "")
        return weather

    def _predict(self, turbine_id: int, issue_time: pd.Timestamp, weather: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._model_metadata = {}
        if self.model_runner is not None:
            outputs = self.model_runner(turbine_id, issue_time, weather.copy(deep=True))
        else:
            from .ml_bridge import run_team_model
            team_result = run_team_model(turbine_id, issue_time, weather.copy(deep=True), self.artifacts_dir)
            self._model_metadata = team_result.attrs["model"]
            outputs = tuple(team_result[name].to_numpy() for name in ("level1", "residual_pred", "prediction"))
        if len(outputs) != 3:
            raise ValueError("Model must return level1, residual_pred, prediction")
        arrays = tuple(np.asarray(value, dtype=float) for value in outputs)
        if any(value.shape != (HORIZON,) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Each model output must be a finite vector of 48 values")
        return arrays

    def _forecast_from(self, turbine_id: int, stamp: pd.Timestamp, weather: pd.DataFrame, trigger: str) -> pd.DataFrame:
        LOG.info("[TOOL] run_forecast_model: turbine %s", turbine_id)
        level1, residual, prediction = self._predict(turbine_id, stamp, weather)
        result = weather[["time", "wind_fc"]].copy()
        result["level1"] = level1
        result["residual_pred"] = residual
        result["prediction"] = prediction
        result["lead_hours"] = np.arange(1, len(result) + 1)
        self.validate_prediction(result, weather)
        LOG.info("[TOOL] validate_prediction: ok")
        result = result[COLUMNS]
        result.attrs = {"weather": {k: v for k, v in weather.attrs.items() if not k.startswith("_")},
                        "model": dict(self._model_metadata)}

        previous = self.latest_published(turbine_id, stamp)
        analysis = self.analyze_forecast(result, weather, previous)
        LOG.info("[TOOL] analyze_forecast: %s", analysis["summary"])
        for flag in analysis["flags"]:
            LOG.info("[AGENT] note: %s", flag["text"])

        inputs = weather[[c for c in ("time", "run_day", "wind_fc", "temperature") if c in weather]].copy()
        inputs["time"] = inputs["time"].map(lambda t: t.isoformat())
        inputs_records = json.loads(inputs.to_json(orient="records"))
        same_issue = previous is not None and pd.Timestamp(previous["issue_time"]) == stamp
        result.attrs["agent"] = {
            "trigger": trigger,
            "version": int(previous.get("version", 1)) + 1 if same_issue else 1,
            "supersedes": previous["run_id"] if previous else None,
            "inputs": inputs_records,
            "inputs_sha256": hashlib.sha256(json.dumps(inputs_records, sort_keys=True).encode()).hexdigest(),
            "analysis": analysis,
        }
        self.save_forecast(turbine_id, stamp, result)
        return result

    def _run_id(self) -> str | None:
        return self.last_saved_path.stem.split("_")[-1] if self.last_saved_path else None

    def _current_model_version(self, turbine_id: int) -> str | None:
        if self.model_runner is not None:
            return None
        manifest = self.artifacts_dir / f"manifest_t{turbine_id}.json"
        return json.loads(manifest.read_text(encoding="utf-8")).get("model_version") if manifest.is_file() else None
