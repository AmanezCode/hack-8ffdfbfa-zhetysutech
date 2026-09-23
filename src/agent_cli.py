"""Run the forecast agent from the command line.

Replay (February 2026, pinned archive):
  python -m src.agent_cli --turbine 1 --issue-time 2026-01-31T17:00:00Z
  python -m src.agent_cli --turbine 1 --issue-time 2026-01-31T17:00:00Z --updates 6,12,18
Real time (live Open-Meteo):
  python -m src.agent_cli --turbine 1 --live
  python -m src.agent_cli --turbine 1 --watch 60          # re-check every 60 min until Ctrl+C

Dispatcher briefing written by an LLM that calls the agent's tools:
  python -m src.agent_cli --turbine 1 --issue-time 2026-02-03T17:00:00Z --brief --llm openai

Issue times are UTC; 17:00Z is 23:00 on the SCADA clock (UTC+6). Logs go to
stderr, a JSON summary of every decision to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from .config import FORECASTS, TURBINES
from .forecast_agent import ForecastAgent
from .llm_agent import PROVIDERS, OperatorAgent
from .model import IncompleteWeatherError


def summarise(decision: dict) -> dict:
    analysis = decision.get("analysis")
    return {"check_time": decision["check_time"], "recomputed": decision["recomputed"], "run_id": decision.get("run_id"),
            "changed_hours": decision.get("changed_hours"), "fresher_runs": decision.get("fresher_runs"),
            "expired": decision.get("expired"), "model_changed": decision.get("model_changed"),
            "summary": analysis["summary"] if analysis else "inputs unchanged, forecast kept"}


def current_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h")


def watch(agent: ForecastAgent, turbine: int, minutes: float, iterations: int) -> None:
    done = 0
    try:
        while True:
            decision = agent.check_for_update(turbine, current_hour(), refresh_weather=True)
            print(json.dumps(summarise(decision), ensure_ascii=False), flush=True)
            done += 1
            if iterations and done >= iterations:
                return
            time.sleep(minutes * 60)
    except KeyboardInterrupt:
        logging.info("watch stopped after %d check(s)", done)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 48-hour wind forecast agent")
    parser.add_argument("--turbine", type=int, choices=sorted(TURBINES), required=True)
    parser.add_argument("--issue-time", help="Aware whole hour, e.g. 2026-01-31T17:00:00Z")
    parser.add_argument("--updates", default="", help="Re-check offsets in hours after the issue, e.g. 6,12,18")
    parser.add_argument("--refresh-weather", action="store_true", help="Re-request Open-Meteo live instead of the pinned archive only")
    parser.add_argument("--live", action="store_true", help="Forecast from the current hour with live Open-Meteo data")
    parser.add_argument("--watch", type=float, metavar="MINUTES", help="Keep checking for fresher weather every MINUTES")
    parser.add_argument("--iterations", type=int, default=0, help="Stop --watch after N checks (0 = until Ctrl+C)")
    parser.add_argument("--brief", action="store_true", help="Write a dispatcher briefing with an LLM that calls the agent's tools")
    parser.add_argument("--llm", choices=PROVIDERS, default="auto",
                        help="auto picks Claude or OpenAI by the API key present, otherwise a template without LLM")
    parser.add_argument("--output", default=str(FORECASTS / "agent"))
    args = parser.parse_args()
    if not (args.issue_time or args.live or args.watch):
        parser.error("give --issue-time, --live or --watch")

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TURBINES[args.turbine]
    agent = ForecastAgent(output_dir=args.output, coordinates={args.turbine: (cfg["lat"], cfg["lon"])})
    checks = tuple(int(h) for h in args.updates.split(",") if h.strip())

    try:
        if args.brief:
            issue = current_hour() if args.live else args.issue_time
            briefing = OperatorAgent(agent, args.llm).brief(args.turbine, issue, checks=checks or (6, 12))
            print(json.dumps({"provider": briefing.provider, "model": briefing.model, "note": briefing.note,
                              "tools_called": [t["tool"] for t in briefing.trace]}, ensure_ascii=False))
            print(briefing.text)
            return 0
        if args.watch:
            watch(agent, args.turbine, args.watch, args.iterations)
            return 0
        issue = current_hour() if args.live else args.issue_time
        refresh = args.refresh_weather or args.live
        if checks:
            decisions = agent.run_update_cycle(args.turbine, issue, checks=checks, refresh_weather=refresh)
        else:
            result = agent.run(args.turbine, issue, refresh_weather=refresh)
            decisions = [{"check_time": pd.Timestamp(issue).isoformat(), "recomputed": True, "run_id": agent._run_id(),
                          "analysis": result.attrs["agent"]["analysis"]}]
    except (ValueError, OSError, IncompleteWeatherError) as exc:
        logging.error("Forecast not produced: %s", exc)
        return 1

    print(json.dumps({"turbine": args.turbine, "runs": [summarise(d) for d in decisions], "output": args.output},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
