# Shared decisions

- Role #2 owns ForecastAgent. Existing forecast_pipeline.py is preserved.
- Backend receives ordered columns: time, level1, residual_pred, prediction,
  wind_fc, lead_hours. JSON envelope holds metadata plus forecast records.
- Horizon uses issue_time + 1..48 hours, UTC. Naive issue times are rejected
  at the agent boundary. This corrects the previous off-by-one output.
- Default weather adapter uses native Previous Runs with original provenance
  and current availability rule (12h delay). Raw forecast.json is not used.
- Team ML source integrated from main 98a9a59; ml_bridge.py is default backend.
  It loads model_tN.joblib, booster_tN.txt, manifest_tN.json and preserves naive
  UTC inside ML. SCADA conversion is fixed UTC+6, never Asia/Almaty.
  Native features select weather over +0..49 hours and output +1..48.
  Generic artifact_adapter.py remains optional and requires metadata_tN.json.
- No new ML training, backtest score claims or dashboard work in role #2 scope.
- User subsequently authorized commit and push to ibrahim after integration.
