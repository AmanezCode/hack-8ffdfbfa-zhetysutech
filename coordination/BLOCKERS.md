# Integration blockers

- ML -> Agents: source integrated from main; trained model_t{1,2}.joblib and
  residual_t{1,2}.txt still absent. Actual team format verified on synthetic test.
- Weather -> Agents: data/raw/forecast.json has no issuance/availability metadata.
  It cannot be used as a historically verified forecast. Need archived vintages
  from a source that preserves model run and actual publication time.
- Configuration: CLI coordinates should match src/config.py from ML team.
- Evaluation: February ground truth is absent; agent tests are not February MAE.
