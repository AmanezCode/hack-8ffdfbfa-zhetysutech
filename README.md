# hack-8ffdfbfa-zhetysutech
Hackathon team repository for ZhetysuTech

## Forecasting baseline

The first runnable baseline is in `src/forecast_pipeline.py`. It reads the two 10-minute turbine files from `data/raw`, aggregates them to hourly observations, trains a leakage-safe gradient boosting model, runs weekly rolling validation on the available pre-test history, and writes a 48-hour forecast.

```powershell
python -m pip install -r requirements.txt
python src/forecast_pipeline.py --horizon 48
```

Outputs are written to `outputs/`: `audit.json`, `backtest_results.csv`, and `forecast.csv`.

## ForecastAgent

Роль #2 реализована в `src/forecast_agent.py`. Агент выполняет загрузку погоды,
проверку 48 часов, вызов ML, проверку результата и сохранение JSON.
Контракт и подключение ML описаны в [docs/FORECAST_AGENT.md](docs/FORECAST_AGENT.md).
Быстрая проверка без внешних данных:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.agent_demo
```

Демо использует только синтетическую погоду и тестовый расчёт мощности;
оно не является историческим прогнозом и не измеряет MAE.
Рабочий CLI: `python -m src.agent_cli --help`.
Результат: DataFrame `[time, level1, residual_pred, prediction, wind_fc, lead_hours]`
и JSON в `forecasts/`, с метаданными запуска и погодного источника.
Время обязательно содержит часовой пояс; горизонт — часы +1..+48.
Для реального запуска нужны артефакты ML, подтверждённые координаты и архивные
прогнозы с датами выпуска/доступности. ML-код из main интегрирован через
src/ml_bridge.py: PowerCurve + LightGBM, без изменения модулей товарища.
Для него архив содержит 50 часов (+0..49), результат — 48 часов (+1..48).
Обученные рабочие артефакты и подтверждённые архивные выпуски пока отсутствуют.

The supplied measurements end at 2026-01-31 23:00. Therefore February 2026 MAE cannot be computed from these files; a historically archived weather forecast vintage and February ground truth are required for final competition validation. `forecast.json` is retained as an input artifact, but is not treated as an archived forecast vintage.
