"""PowerShell: python -m src.agent_cli --help."""
from __future__ import annotations

import argparse
import json
import logging

from .forecast_agent import ForecastAgent, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a validated 48-hour turbine forecast")
    parser.add_argument("--turbine", type=int, choices=(1, 2), required=True)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--issue-time", required=True, help="Aware whole hour, e.g. 2026-01-31T17:00:00Z (23:00 SCADA UTC+6)")
    parser.add_argument("--weather", help="Flat vintage JSON for generic backend; team default uses Previous Runs cache")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--model-backend", choices=("team", "generic"), default="team")
    parser.add_argument("--output", default=str(ROOT / "forecasts"))
    args = parser.parse_args()
    from .config import TURBINES
    if (args.lat is None) != (args.lon is None):
        parser.error("Provide both --lat and --lon, or neither to use src.config")
    if args.model_backend == "team" and args.weather:
        parser.error("Team model requires Previous Runs cache; --weather is for --model-backend generic")
    cfg = TURBINES[args.turbine]
    coordinates = (cfg["lat"], cfg["lon"]) if args.lat is None else (args.lat, args.lon)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    agent = ForecastAgent(output_dir=args.output, artifacts_dir=args.artifacts, model_backend=args.model_backend,
                          coordinates={args.turbine: coordinates}, archive_path=args.weather)
    try:
        result = agent.run(args.turbine, args.issue_time)
    except (ValueError, OSError, ImportError, TypeError, RuntimeError) as exc:
        logging.error("Forecast not produced: %s", exc)
        return 1
    print(json.dumps({"rows": len(result), "path": str(agent.last_saved_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
