"""Operator layer: an LLM plans the forecasting work, calls the agent's tools and
writes the dispatcher briefing.

Providers:
  claude   Anthropic SDK tool runner, model claude-opus-5 (ANTHROPIC_API_KEY)
  openai   OpenAI SDK Chat Completions, model from OPENAI_MODEL (OPENAI_API_KEY;
           OPENAI_BASE_URL also reaches OpenAI-compatible endpoints such as NVIDIA NIM)
  template no LLM: the same tools run in a fixed order and a template fills the text

Every number the briefing may contain comes from a tool, and the tools run the
real pipeline (weather available at issue time -> ML -> validation -> analysis).
If a provider is unavailable, refuses or errors, the template path answers so
the demo never depends on an API key.
"""
from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import dataclass, field

import pandas as pd

from src.config import SCADA_UTC_OFFSET_HOURS
from src.forecast_agent import ForecastAgent, issue_timestamp

LOG = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
MAX_TOOL_ROUNDS = 8
PROVIDERS = ("auto", "claude", "openai", "template")

SYSTEM_PROMPT = """You are the forecasting operator of a two-turbine wind farm in Kazakhstan and you brief the grid dispatcher in Russian.

Your tools run the real forecasting pipeline. Every number you state must come from a tool result in this conversation; if a tool fails, say so instead of estimating.

Power is normalised: a share of rated capacity (0-1), report it in %. All times are on the SCADA clock (UTC+6).

The briefing covers expected production per day (capacity factor, full-load hours), peaks and ramps with their times, how uncertain the forecast is (80% interval), what changed against the previous forecast and after later weather updates, and concrete actions for the dispatcher (reserve around ramps, hours to double-check). A dispatcher should be able to read it in a minute, so leave out tool names, field names and run ids."""


@dataclass
class Briefing:
    text: str
    provider: str
    model: str | None
    trace: list[dict] = field(default_factory=list)
    note: str | None = None


def to_scada(stamp: pd.Timestamp) -> pd.Timestamp:
    return stamp.tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=SCADA_UTC_OFFSET_HOURS)


def scada_iso(stamp: pd.Timestamp) -> str:
    return stamp.tz_convert(f"Etc/GMT-{SCADA_UTC_OFFSET_HOURS}").isoformat()


class OperatorAgent:
    def __init__(self, agent: ForecastAgent, provider: str = "auto", *, anthropic_client=None, openai_client=None):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        self.agent = agent
        self.provider = provider
        self.anthropic_client = anthropic_client
        self.openai_client = openai_client
        self.trace: list[dict] = []

    # ----------------------------------------------------------------- tools

    def forecast_power(self, turbine_id: int, issue_time: str) -> str:
        """Run the full forecast cycle for one turbine and return the 48-hour forecast with its analysis.

        Uses only weather published before issue_time, runs the ML model, validates and saves the result.
        Returns hourly power with the calibrated 80% interval, per-day energy, ramps, flags and the revision
        against the previously published forecast.

        Args:
            turbine_id: Turbine number, 1 or 2.
            issue_time: Issue time in ISO 8601 with offset, e.g. 2026-02-03T23:00:00+06:00 (SCADA clock).
        """
        stamp = issue_timestamp(issue_time)
        result = self.agent.run(turbine_id, stamp)
        analysis = result.attrs["agent"]["analysis"]
        interval = analysis.get("interval_80") or {}
        card = result.attrs.get("model", {})
        return json.dumps({
            "turbine_id": turbine_id,
            "issue_time_scada": str(to_scada(stamp)),
            "run_id": self.agent._run_id(),
            "model_version": card.get("model_version"),
            "units": "power as share of rated capacity (0-1); times on the SCADA clock (UTC+6)",
            "hourly": {
                "first_hour_scada": str(to_scada(result["time"].iloc[0])),
                "power": [round(float(v), 3) for v in result["prediction"]],
                "p10": [round(float(v), 3) for v in interval.get("low", [])],
                "p90": [round(float(v), 3) for v in interval.get("high", [])],
            },
            "days": analysis["days"],
            "ramps": analysis["ramps"],
            "flags": [flag["text"] for flag in analysis["flags"]],
            "revision_vs_previous": analysis["revision"],
            "interval_cv_coverage": interval.get("cv_coverage"),
        }, ensure_ascii=False)

    def check_for_update(self, turbine_id: int, check_time: str) -> str:
        """Check whether fresher weather runs are available at check_time and recompute the forecast if so.

        The agent recomputes when the admissible weather for already forecast hours changed, when the model was
        retrained, or when the published forecast covers less than 24 hours ahead; otherwise it keeps it.

        Args:
            turbine_id: Turbine number, 1 or 2.
            check_time: Time of the check in ISO 8601 with offset, e.g. 2026-02-04T05:00:00+06:00 (SCADA clock).
        """
        decision = self.agent.check_for_update(turbine_id, issue_timestamp(check_time))
        analysis = decision.pop("analysis", None)
        if analysis:
            decision["summary"] = analysis["summary"]
            decision["revision_vs_previous"] = analysis["revision"]
            decision["days"] = analysis["days"]
        return json.dumps(decision, ensure_ascii=False, default=str)

    def model_quality(self, turbine_id: int) -> str:
        """Return the model's measured quality from rolling validation (November 2025 - January 2026).

        Includes accuracy (1 - MAE as share of capacity), share of hours within 10% of capacity, error by
        horizon, skill against simple baselines, gain from re-issuing with fresher weather and the coverage of
        the 80% interval.

        Args:
            turbine_id: Turbine number, 1 or 2.
        """
        manifest = json.loads((self.agent.artifacts_dir / f"manifest_t{turbine_id}.json").read_text(encoding="utf-8"))
        chosen = manifest["variant"]
        cv = manifest["cv_mean"]
        folds = manifest["cv_folds"]
        by_lead = {label: round(sum(f["mae_by_lead"][label][chosen] for f in folds) / len(folds), 4) for label in ("1-24h", "25-48h")}
        return json.dumps({
            "model_version": manifest["model_version"],
            "accuracy": round(cv[chosen]["accuracy"], 4),
            "mae": round(cv[chosen]["MAE"], 4),
            "within_10pct_of_capacity": round(cv[chosen]["within_10pct"], 4),
            "mae_by_horizon": by_lead,
            "daily_energy_error": round(cv[chosen]["daily_energy_error"], 4),
            "skill_vs": {ref: round(1 - cv[chosen]["MAE"] / cv[ref]["MAE"], 4) for ref in ("physics", "climatology", "persistence")},
            "reissue_gain": {k: round(v["improvement"], 4) for k, v in manifest.get("update_gain", {}).items()},
            "interval_80_cv_coverage": (manifest.get("interval") or {}).get("cv_coverage"),
            "note": "validated on Nov 2025 - Jan 2026; February actuals are not published",
        }, ensure_ascii=False)

    def _traced(self, func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            try:
                result, status = func(**kwargs), "ok"
            except Exception as exc:  # returned to the model as a tool error; it must not invent the missing numbers
                result, status = json.dumps({"error": f"{type(exc).__name__}: {exc}"}), "error"
                LOG.warning("[LLM] tool %s failed: %s", func.__name__, exc)
            self.trace.append({"tool": func.__name__, "input": kwargs, "status": status})
            LOG.info("[LLM] tool %s(%s) -> %s", func.__name__, ", ".join(f"{k}={v}" for k, v in kwargs.items()), status)
            return result
        return wrapper

    def _tool_functions(self) -> list:
        return [self._traced(f) for f in (self.forecast_power, self.check_for_update, self.model_quality)]

    # ------------------------------------------------------------- briefing

    def resolve_provider(self) -> str:
        if self.provider != "auto":
            return self.provider
        if self.anthropic_client is not None or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "claude"
        if self.openai_client is not None or os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "template"

    def brief(self, turbine_id: int, issue_time, checks: tuple[int, ...] = (6, 12), question: str | None = None) -> Briefing:
        self.trace = []
        stamp = issue_timestamp(issue_time)
        provider = self.resolve_provider()
        request = (f"Турбина {turbine_id}. Плановый выпуск прогноза: issue_time={scada_iso(stamp)} (часы SCADA). "
                   f"Подготовь брифинг диспетчеру.")
        if checks:
            times = ", ".join(scada_iso(stamp + pd.Timedelta(hours=h)) for h in checks)
            request += f" Затем проверь обновления прогноза в моменты {times} и скажи, что изменилось."
        if question:
            request += f" Вопрос диспетчера: {question}"

        if provider in ("claude", "openai"):
            try:
                text, model = self._run_claude(request) if provider == "claude" else self._run_openai(request)
                return Briefing(text=text, provider=provider, model=model, trace=list(self.trace))
            except _ProviderUnavailable as exc:
                note = f"{provider} недоступен ({exc}); брифинг собран по шаблону"
                LOG.warning("[LLM] %s", note)
                self.trace = []
                return Briefing(text=self._template(turbine_id, stamp, checks), provider="template", model=None,
                                trace=list(self.trace), note=note)
        return Briefing(text=self._template(turbine_id, stamp, checks), provider="template", model=None, trace=list(self.trace))

    def _run_claude(self, request: str) -> tuple[str, str]:
        import anthropic
        from anthropic import beta_tool

        try:
            client = self.anthropic_client or anthropic.Anthropic()
            runner = client.beta.messages.tool_runner(
                model=CLAUDE_MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=[beta_tool(f) for f in self._tool_functions()],
                messages=[{"role": "user", "content": request}],
                # If Claude declines, the API re-runs the request on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                max_iterations=MAX_TOOL_ROUNDS,
            )
            final = None
            for message in runner:
                final = message
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise _ProviderUnavailable(f"no valid credentials: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise _ProviderUnavailable(f"no connection: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise _ProviderUnavailable(f"API error {exc.status_code}") from exc
        except TypeError as exc:  # raised by the SDK when no credential source resolves
            if "authentication" not in str(exc).lower():
                raise
            raise _ProviderUnavailable("no API key or profile") from exc

        if final is None or final.stop_reason == "refusal":
            raise _ProviderUnavailable("the model declined the request")
        text = "\n".join(block.text for block in final.content if block.type == "text").strip()
        if not text:
            raise _ProviderUnavailable(f"no text in the final answer (stop_reason={final.stop_reason})")
        return text, final.model

    def _run_openai(self, request: str) -> tuple[str, str]:
        import openai
        from anthropic import beta_tool

        functions = {f.__name__: f for f in self._tool_functions()}
        tools = [{"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                                   "parameters": spec["input_schema"]}}
                 for spec in (beta_tool(f).to_dict() for f in functions.values())]
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]

        try:
            client = self.openai_client or openai.OpenAI()
            for _ in range(MAX_TOOL_ROUNDS):
                response = client.chat.completions.create(model=model, messages=messages, tools=tools)
                message = response.choices[0].message
                if message.refusal:
                    raise _ProviderUnavailable("the model declined the request")
                if not message.tool_calls:
                    if not (message.content or "").strip():
                        raise _ProviderUnavailable("empty answer")
                    return message.content.strip(), response.model
                messages.append({"role": "assistant", "content": message.content,
                                 "tool_calls": [call.model_dump() for call in message.tool_calls]})
                for call in message.tool_calls:
                    handler = functions.get(call.function.name)
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = None
                    if handler is None or arguments is None:
                        output = json.dumps({"error": f"unknown tool or malformed arguments: {call.function.name}"})
                    else:
                        output = handler(**arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise _ProviderUnavailable(f"no valid credentials: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise _ProviderUnavailable(f"no connection: {exc}") from exc
        except openai.APIStatusError as exc:
            raise _ProviderUnavailable(f"API error {exc.status_code}") from exc
        except openai.OpenAIError as exc:  # e.g. no API key configured
            raise _ProviderUnavailable(str(exc)) from exc
        raise _ProviderUnavailable(f"no final answer after {MAX_TOOL_ROUNDS} tool rounds")

    def _template(self, turbine_id: int, stamp: pd.Timestamp, checks: tuple[int, ...]) -> str:
        tools = {f.__name__: f for f in self._tool_functions()}
        quality = json.loads(tools["model_quality"](turbine_id=turbine_id))
        forecast = json.loads(tools["forecast_power"](turbine_id=turbine_id, issue_time=stamp.isoformat()))
        if "error" in forecast:
            return f"Прогноз не построен: {forecast['error']}"

        lines = [f"Брифинг: турбина {turbine_id}, выпуск {to_scada(stamp):%d.%m %H:%M} (SCADA), модель {forecast['model_version']}.", ""]
        for day in forecast["days"]:
            if day["hours"] < 12:
                continue
            lines.append(f"- {pd.Timestamp(day['date_scada']):%d.%m}: КИУМ {day['capacity_factor']:.0%}, "
                         f"{day['full_load_hours']:.1f} ч полной мощности, пик {day['peak_power']:.0%} в {day['peak_scada'][-5:]}"
                         + (f", ожидаемая ошибка ±{day['expected_mae']:.0%}" if day.get("expected_mae") else "") + ".")
        for ramp in forecast["ramps"]:
            lines.append(f"- Рампа {'вверх' if ramp['direction'] == 'up' else 'вниз'} {ramp['change']:+.0%} за 3 ч "
                         f"с {pd.Timestamp(ramp['start_scada']):%d.%m %H:%M}.")
        low, high = forecast["hourly"]["p10"], forecast["hourly"]["p90"]
        if low:
            width = sum(h - l for l, h in zip(low, high)) / len(low)
            lines.append(f"- Неопределённость: 80%-ный интервал в среднем шириной {width:.0%} номинала"
                         + (f" (на проверке покрыл {forecast['interval_cv_coverage']:.0%} фактов)." if forecast.get("interval_cv_coverage") else "."))
        revision = forecast["revision_vs_previous"]
        if revision:
            lines.append(f"- К прошлому прогнозу: изменение в среднем {revision['mean_abs_change']:.1%}, "
                         f"максимум {revision['max_abs_change']:.0%} в {revision['max_change_scada'][-11:]}.")
        for hours in checks:
            update = json.loads(tools["check_for_update"](turbine_id=turbine_id,
                                                          check_time=(stamp + pd.Timedelta(hours=hours)).isoformat()))
            if "error" in update:
                lines.append(f"- +{hours} ч: проверка не выполнена ({update['error']}).")
            elif update["recomputed"]:
                why = "устарел" if update.get("expired") else f"свежая погода для {update['changed_hours']} ч"
                rev = update.get("revision_vs_previous") or {}
                lines.append(f"- +{hours} ч: {why} — прогноз пересчитан"
                             + (f", изменение в среднем {rev['mean_abs_change']:.1%}." if rev else "."))
            else:
                lines.append(f"- +{hours} ч: входные данные не изменились, прогноз оставлен.")
        lines.append(f"- Качество (ноябрь–январь): точность {quality['accuracy']:.1%}, "
                     f"{quality['within_10pct_of_capacity']:.0%} часов в пределах ±10% номинала, "
                     f"лучше климатологии на {quality['skill_vs']['climatology']:.0%}.")

        actions = []
        if forecast["ramps"]:
            actions.append("держать резерв регулирования около рамп")
        if low and width > 0.5:
            actions.append("для обязательств планировать по нижней границе интервала (p10)")
        if forecast["flags"]:
            actions.append("проверить отмеченные часы: " + "; ".join(forecast["flags"]))
        if actions:
            lines.append("\n**Рекомендации:** " + "; ".join(actions) + ".")
        return "\n".join(lines)


class _ProviderUnavailable(RuntimeError):
    pass
