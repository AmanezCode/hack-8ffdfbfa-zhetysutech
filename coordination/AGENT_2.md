# AGENT_2 — Agents

Status: IMPLEMENTED AND VERIFIED against main 98a9a59
Branch: ibrahim
Scope: ForecastAgent, CLI, integration tests, documentation.
Delegated disjoint scopes: Kepler — weather.py and weather tests;
Meitner — model.py and model tests.
Preserve forecast_pipeline.py and existing uncommitted baseline work.
User now authorized sync, commit and push to ibrahim.
Integrated origin/main 98a9a59. ml_bridge uses native Previous Runs and naive UTC.
Team model loader fixes CRLF parsing on Windows; no changes to training logic.
Validation: 45 tests, 26 ML acceptance checks and all 56 agent replay issues
(2688 rows) matching committed ML reference. No tracked forecast overwritten.

Contract: time, level1, residual_pred, prediction, wind_fc, lead_hours.
All times timezone-aware UTC; valid times issue_time + 1..48 hours.
Weather input must identify source, issuance and availability.
ML artifact formats and turbine coordinates must be explicit, never guessed.

Implemented: five tools, run/forecast orchestration, bounded transient retries,
UTC validation, versioned atomic JSON output, CLI, synthetic demo and tests.
Worker review: weather adapter and nine tests (Kepler), model adapter and nine
tests (Meitner), orchestration tests and stale-result regression (Tesla).
Parent integration test uses real serialized sklearn estimator via CLI.
See docs/FORECAST_AGENT.md and BLOCKERS.md for handoff and remaining inputs.
