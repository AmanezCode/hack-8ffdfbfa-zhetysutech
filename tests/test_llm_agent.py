"""Operator LLM layer: provider wiring, tool calls and the template fallback, without network or keys."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import openai
from openai.types.chat import ChatCompletion

from src.config import TURBINES
from src.forecast_agent import ForecastAgent
from src.llm_agent import CLAUDE_MODEL, SYSTEM_PROMPT, OperatorAgent

ISSUE = "2026-02-03T17:00:00Z"
NO_KEYS = {k: "" for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")}


class FakeRunner:
    def __init__(self, kwargs, final_stop):
        self.kwargs, self.final_stop = kwargs, final_stop

    def __iter__(self):
        tools = {tool.name: tool for tool in self.kwargs["tools"]}
        tools["forecast_power"].call({"turbine_id": 1, "issue_time": "2026-02-03T23:00:00+06:00"})
        yield SimpleNamespace(stop_reason="tool_use", model=CLAUDE_MODEL, content=[])
        yield SimpleNamespace(stop_reason=self.final_stop, model=CLAUDE_MODEL,
                              content=[SimpleNamespace(type="text", text="Брифинг от Claude")])


class FakeAnthropic:
    def __init__(self, final_stop="end_turn"):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._runner))
        self.final_stop = final_stop

    def _runner(self, **kwargs):
        self.calls.append(kwargs)
        return FakeRunner(kwargs, self.final_stop)


def completion(message: dict, finish: str) -> ChatCompletion:
    return ChatCompletion.model_validate({"id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4.1",
                                         "choices": [{"index": 0, "finish_reason": finish, "message": message}]})


class FakeOpenAI:
    def __init__(self, error=None):
        self.sent = []
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self.error:
            raise self.error
        self.sent.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        if len(self.sent) == 1:
            return completion({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "model_quality", "arguments": "{\"turbine_id\": 1}"}},
                {"id": "call_2", "type": "function", "function": {"name": "forecast_power",
                                                                  "arguments": json.dumps({"turbine_id": 1, "issue_time": ISSUE})}},
            ]}, "tool_calls")
        return completion({"role": "assistant", "content": "Брифинг от OpenAI"}, "stop")


class OperatorAgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cfg = {t: (c["lat"], c["lon"]) for t, c in TURBINES.items()}
        self.agent = ForecastAgent(output_dir=Path(directory.name), coordinates=cfg)

    def test_template_briefing_uses_real_tools_without_keys(self):
        with patch.dict(os.environ, NO_KEYS):
            briefing = OperatorAgent(self.agent).brief(1, ISSUE, checks=(6,))
        self.assertEqual(briefing.provider, "template")
        self.assertEqual([t["tool"] for t in briefing.trace], ["model_quality", "forecast_power", "check_for_update"])
        self.assertIn("Брифинг: турбина 1, выпуск 03.02 23:00", briefing.text)
        self.assertIn("+6 ч", briefing.text)

    def test_claude_runner_gets_model_fallbacks_and_tools(self):
        client = FakeAnthropic()
        briefing = OperatorAgent(self.agent, "claude", anthropic_client=client).brief(1, ISSUE, checks=(6,))
        call = client.calls[0]
        self.assertEqual(call["model"], "claude-opus-5")
        self.assertEqual(call["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(call["fallbacks"], "default")
        self.assertEqual(call["system"], SYSTEM_PROMPT)
        self.assertEqual({t.name for t in call["tools"]}, {"forecast_power", "check_for_update", "model_quality"})
        self.assertIn("issue_time=2026-02-03T23:00:00+06:00", call["messages"][0]["content"])
        self.assertEqual((briefing.provider, briefing.text), ("claude", "Брифинг от Claude"))
        self.assertEqual(briefing.trace[0]["status"], "ok")
        self.assertEqual(len(list(self.agent.output_dir.glob("*.json"))), 1)

    def test_claude_refusal_falls_back_to_template(self):
        with self.assertLogs("src.llm_agent", level="WARNING"):
            briefing = OperatorAgent(self.agent, "claude", anthropic_client=FakeAnthropic("refusal")).brief(1, ISSUE, checks=())
        self.assertEqual(briefing.provider, "template")
        self.assertIn("declined", briefing.note)
        self.assertIn("Брифинг: турбина 1", briefing.text)

    def test_openai_tool_loop_returns_results_to_the_model(self):
        client = FakeOpenAI()
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4.1"}):
            briefing = OperatorAgent(self.agent, "openai", openai_client=client).brief(1, ISSUE, checks=())
        self.assertEqual((briefing.provider, briefing.text, briefing.model), ("openai", "Брифинг от OpenAI", "gpt-4.1"))
        second_request = client.sent[1]
        tool_messages = [m for m in second_request if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_messages], ["call_1", "call_2"])
        self.assertEqual(json.loads(tool_messages[0]["content"])["accuracy"] > 0.8, True)
        self.assertEqual(len(json.loads(tool_messages[1]["content"])["hourly"]["power"]), 48)
        self.assertEqual([t["tool"] for t in briefing.trace], ["model_quality", "forecast_power"])

    def test_openai_error_falls_back_to_template(self):
        client = FakeOpenAI(error=openai.OpenAIError("The api_key client option must be set"))
        with self.assertLogs("src.llm_agent", level="WARNING"):
            briefing = OperatorAgent(self.agent, "openai", openai_client=client).brief(1, ISSUE, checks=())
        self.assertEqual(briefing.provider, "template")
        self.assertIn("api_key", briefing.note)

    def test_auto_prefers_available_provider(self):
        with patch.dict(os.environ, NO_KEYS):
            self.assertEqual(OperatorAgent(self.agent).resolve_provider(), "template")
        with patch.dict(os.environ, {**NO_KEYS, "OPENAI_API_KEY": "sk-test"}):
            self.assertEqual(OperatorAgent(self.agent).resolve_provider(), "openai")
        with patch.dict(os.environ, {**NO_KEYS, "ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-test"}):
            self.assertEqual(OperatorAgent(self.agent).resolve_provider(), "claude")

    def test_tool_errors_are_returned_not_raised(self):
        operator = OperatorAgent(self.agent)
        forecast = {f.__name__: f for f in operator._tool_functions()}["forecast_power"]
        payload = json.loads(forecast(turbine_id=3, issue_time=ISSUE))
        self.assertIn("error", payload)
        self.assertEqual(operator.trace[-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
