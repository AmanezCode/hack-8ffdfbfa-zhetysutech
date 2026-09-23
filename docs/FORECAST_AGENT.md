# ForecastAgent: передача backend-разработчику

## Что готово

`ForecastAgent` — детерминированный агент-оркестратор. LLM и платные API не нужны.
Пять инструментов доступны как методы класса с именами из контракта роли #2:

| Метод | Результат |
|---|---|
| get_archived_weather_forecast(lat, lon, issue_date) | DataFrame погоды |
| validate_weather(weather) | None или ValueError |
| run_forecast_model(turbine_id, issue_time) | level1, residual_pred, prediction |
| validate_prediction(pred, weather) | None или ValueError |
| save_forecast(turbine_id, issue_time, pred) | Path к JSON |

`run(turbine_id, issue_time)` выполняет весь цикл. При TimeoutError/ConnectionError
повторяет загрузку и расчёт (максимум 2 попытки по умолчанию). Некорректные данные
не исправляет вымышленными значениями. Повторный вызов после обновления входов
перечитывает погоду/модели и сохраняет новую версию. Фонового планировщика нет:
backend должен вызывать run при обновлении данных или по своему расписанию.

## Время и проверки

`issue_time` — целый час с явным offset, например `2026-01-31T23:00:00+05:00`.
Внутри время преобразуется в UTC. Каждая строка относится к issue_time + lead_hours;
lead_hours строго 1..48. Значения погоды: wind_fc в м/с, temperature в °C.
Проверяются пропуски, бесконечности, дубликаты, часовой шаг, совпадение времени
погоды и результата, диапазон prediction [0,1] и
`prediction = clip(level1 + residual_pred, 0, 1)`.
При нулевом ветре prediction > 0.05 считается ошибкой; отличие от level1 > 0.3
логируется как предупреждение. Это инженерные пороги, не настройка по MAE.

## Использование backend

```python
from src.forecast_agent import ForecastAgent

# Координаты передаёт конфигурация backend; значения не следует угадывать.
agent = ForecastAgent(coordinates={1: (verified_lat, verified_lon)},
                      archive_path="data/weather/vintage.json")
frame = agent.run(1, "2026-01-31T23:00:00+05:00")
records_json = frame.to_json(orient="records", date_format="iso")
saved_path = agent.last_saved_path
```

JSON-файл содержит turbine_id, issue_time, run_id, weather и forecast.
`forecast` — список строк по контракту backend; метаданные weather сохраняют
происхождение входов. Запись через временный файл и rename, имена уникальны.
Логи идут через стандартный Python logging; CLI выводит их в stderr.
Для параллельных запросов используйте отдельный экземпляр агента на запрос.

## Архив погоды

Загрузчик агента находится в src/archived_weather.py. src/weather.py сохранён
из main для ML-команды; он не предоставляет проверяемое время выпуска.
Локальный вход — JSON с полями source, issued_at, available_at, latitude,
longitude, wind_unit и hourly. В hourly нужны time, wind_fc, temperature.
Все даты содержат часовой пояс. issued_at и available_at не позже issue_time.
Для ML-команды нужны 50 значений issue+0..49: два крайних часа используются
для соседних погодных признаков; наружу возвращаются 48 часов issue+1..48.
Для generic-адаптера достаточно 48. Недостающие граничные часы не выдумываются.
Поставщик/импортёр обязан сохранить реальные сведения о публикации; наличие
полей само по себе не доказывает достоверность. Нельзя просто проставить даты
в имеющемся forecast.json: это не сделает его архивным прогнозом.

Без archive_path загрузчик ищет подходящие JSON в data/weather.
Получение реальных прогнозов из внешнего источника остаётся задачей поставщика
погоды; текущий адаптер читает локальные архивы. Можно передать weather_loader
как callable(lat, lon, issue_time), возвращающий DataFrame с метаданными attrs.

## Подключение функций товарища

Исходные src.weather/src.model получены из main (37b1fa2). По умолчанию
src/ml_bridge.py вызывает оригинальную src.model.run_forecast_model и использует
bundle PowerCurve/blend_weight в model_tN.joblib плюс LightGBM residual_tN.txt.
metadata_tN.json для этого формата не нужен. Времена преобразуются из UTC в
Asia/Almaty для соответствия обучающим признакам, затем результат обратно в UTC.
SCADA читается из turbineN.csv либо turbine_N.csv без заполнения пропусков будущими
значениями. История ограничивается issue_time перед передачей модели.
Координаты команды указаны в src/config.py; CLI принимает их явно.
Обученные рабочие артефакты ещё нужны от ML-команды.
Для другого контракта передайте model_runner(turbine_id, issue_time,
weather), который вызывает его функцию и возвращает три массива длиной 48.
Существующую ML-функцию при этом переписывать не нужно.

Опциональный универсальный адаптер перенесён в src/artifact_adapter.py;
выбор: model_backend="generic" или CLI --model-backend generic.
Только для него рядом с каждой парой model_tN.joblib/residual_tN.txt
нужен metadata_tN.json, например (только если residual действительно число):

```json
{"feature_names": ["wind_fc", "temperature"], "residual": {"format": "scalar"}}
```

feature_names можно опустить, если estimator хранит feature_names_in_.
Если указаны оба, их порядок обязан совпадать. Для residual-модели LightGBM:

```json
{"residual": {"format": "lightgbm", "feature_names": ["wind_fc", "temperature"]}}
```

LightGBM входит в requirements.txt команды. Порядок признаков generic-модели
сверяется с самой моделью. test_team_ml.py проверяет настоящий формат команды
через save_artifacts/load_artifacts и расчёт LightGBM на синтетическом примере.
Дополнительные признаки должны быть подготовлены weather_loader; встроенный
адаптер выводит из времени только hour, day_of_year и lead_hours в UTC.
Joblib загружайте только из доверенных файлов ML-команды.

## Команды PowerShell

```powershell
cd C:\Users\user\hack-8ffdfbfa-zhetysutech
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m src.agent_demo
python -m src.agent_cli --help
```

После получения реальных входов (переменные координат задайте подтверждёнными значениями):

```powershell
python -m src.agent_cli --turbine 1 --lat $turbineLat --lon $turbineLon `
  --issue-time '2026-01-31T23:00:00+05:00' `
  --weather 'data/weather/vintage.json' --artifacts artifacts --output forecasts
```

Тесты и демо проверяют программную связку, не качество ML. Существующий baseline
и его январские метрики не подтверждают качество 48-часового NWP-прогноза за февраль.
Команды npm/Capacitor/Electron неприменимы: этот репозиторий содержит Python.
