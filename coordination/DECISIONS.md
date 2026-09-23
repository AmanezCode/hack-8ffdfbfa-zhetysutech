# Shared decisions

- Role #2 owns ForecastAgent. Existing forecast_pipeline.py is preserved.
- Backend receives ordered columns: time, level1, residual_pred, prediction,
  wind_fc, lead_hours. JSON envelope holds metadata plus forecast records.
- Horizon uses issue_time + 1..48 hours, UTC. Naive issue times are rejected
  at the agent boundary. This corrects the previous off-by-one output.
- Weather adapter requires declared issuance/availability and coordinates;
  raw forecast.json is not an eligible archive. Source claims are not verified
  externally. A real provider adapter is still needed for automatic download.
- Team ML source integrated from main; ml_bridge.py is the default backend.
  It preserves team artifact format without metadata_tN.json and converts UTC
  to Asia/Almaty for feature construction. It requires 50 input weather hours.
  Generic artifact_adapter.py remains optional and requires metadata_tN.json.
- No new ML training, backtest score claims or dashboard work in role #2 scope.
- User subsequently authorized commit and push to ibrahim after integration.
