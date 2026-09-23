# Integration blockers

- ML -> Agents: resolved with 98a9a59; trained bundles, boosters and manifests
  integrated and verified on all February issues.
- Weather -> Agents: resolved using committed Previous Runs and native snapshot
  selection. Historical 12h publication delay remains an assumption (docs/ml.md).
  data/raw/forecast.json is not used as an eligible archive.
- Configuration: CLI coordinates should match src/config.py from ML team.
- Evaluation: February ground truth is absent; agent tests are not February MAE.
