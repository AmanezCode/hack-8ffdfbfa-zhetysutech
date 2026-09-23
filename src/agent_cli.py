"""Run the forecast agent from the command line.

python -m src.agent_cli --turbine 1 --issue-time 2026-01-31T17:00:00Z
python -m src.agent_cli --turbine 1 --issue-time 2026-01-31T17:00:00Z --updates 6,12,18
python -m src.agent_cli --turbine 1 --issue-time 2026-01-31T17:00:00Z --refresh-weather

Issue times are UTC; 17:00Z is 23:00 on the SCADA clock (UTC+6). Logs go to
stderr, a JSON summary of every decision to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging

from .config import FORECASTS, TURBINES
from .forecast_agent import ForecastAgent
from .model import IncompleteWeatherError


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 48-hour wind forecast agent")
    parser.add_argument("--turbine", type=int, choices=sorted(TURBINES), required=True)
    parser.add_argument("--issue-time", required=True, help="Aware whole hour, e.g. 2026-01-31T17:00:00Z")
    parser.add_argument("--updates", default="", help="Re-check offsets in hours after the issue, e.g. 6,12,18")
    parser.add_argument("--refresh-weather", action="store_true", help="Re-request Open-Meteo live instead of the pinned archive only")
    parser.add_argument("--output", default=str(FORECASTS / "agent"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TURBINES[args.turbine]
    agent = ForecastAgent(output_dir=args.output, coordinates={args.turbine: (cfg["lat"], cfg["lon"])})
    checks = tuple(int(h) for h in args.updates.split(",") if h.strip())

    try:
        if checks:
            decisions = agent.run_update_cycle(args.turbine, args.issue_time, checks=checks, refresh_weather=args.refresh_weather)
        else:
            result = agent.run(args.turbine, args.issue_time, refresh_weather=args.refresh_weather)
            decisions = [{"check_time": args.issue_time, "recomputed": True, "run_id": agent._run_id(),
                          "analysis": result.attrs["agent"]["analysis"]}]
    except (ValueError, OSError, IncompleteWeatherError) as exc:
        logging.error("Forecast not produced: %s", exc)
        return 1

    runs = [{"check_time": d["check_time"], "recomputed": d["recomputed"], "run_id": d.get("run_id"),
             "changed_hours": d.get("changed_hours"), "fresher_runs": d.get("fresher_runs"),
             "summary": d["analysis"]["summary"] if d.get("analysis") else "inputs unchanged, forecast kept"}
            for d in decisions]
    print(json.dumps({"turbine": args.turbine, "runs": runs, "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
