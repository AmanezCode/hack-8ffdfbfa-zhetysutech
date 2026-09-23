# ForecastAgent: актуальный контракт после ML-коммита 98a9a59

## Запуск

```powershell
cd C:\Users\user\hack-8ffdfbfa-zhetysutech
python -m pip install -r requirements.txt
python -m src.agent_cli --turbine 1 --issue-time '2026-01-31T17:00:00Z'
python -m src.agent_cli --turbine 2 --issue-time '2026-02-27T17:00:00Z'
```

Координаты по умолчанию берутся из src/config.py. Все рабочие модели, история
SCADA и кэш Previous Runs получены из main, запуск возможен без сети.
При отсутствии кэша src.weather.load_weather скачивает его через API.
Результат CLI — путь к JSON; логи идут в stderr.
Демо: `python -m src.agent_demo` (синтетическое, не оценка качества ML).

## Пять инструментов

| Метод | Результат |
|---|---|
| get_archived_weather_forecast(lat, lon, issue_date) | 48 строк погоды, timezone-aware UTC |
| validate_weather(weather) | None или ValueError |
| run_forecast_model(turbine_id, issue_time) | три массива: level1, residual_pred, prediction |
| validate_prediction(pred, weather) | None или ValueError |
| save_forecast(turbine_id, issue_time, pred) | Path к JSON |

`run` выполняет полный цикл. При TimeoutError/ConnectionError повторяет его
до двух раз. Если все допустимые погодные запуски отсутствуют, передаёт
IncompleteWeatherError с missing_hours вызывающему backend без сохранения
неполного результата. Автоматический выбор более старого допустимого запуска
реализован в ML weather_snapshot; неизвестные часы не заполняются.

Повторный вызов перечитывает входы и сохраняет отдельную версию. Планировщик
должен вызывать run при обновлении данных; постоянного фонового процесса нет.
Для одновременных запросов используйте разные экземпляры агента.

## Backend

```python
from src.config import TURBINES
from src.forecast_agent import ForecastAgent

coords = {t: (cfg["lat"], cfg["lon"]) for t, cfg in TURBINES.items()}
agent = ForecastAgent(coordinates=coords)
frame = agent.run(1, "2026-01-31T17:00:00Z")
records_json = frame.to_json(orient="records", date_format="iso")
path = agent.last_saved_path
```

Колонки DataFrame не изменены:
time, level1, residual_pred, prediction, wind_fc, lead_hours.
JSON содержит turbine_id, issue_time, run_id, model, weather, forecast.
model хранит model_version, variant и train_target_end.
weather хранит происхождение API, применённую задержку публикации и run_day
для каждого часа. Внутренняя таблица предыдущих запусков в JSON не сериализуется.
Файл записывается атомарно через временный файл; каждый запуск имеет уникальное имя.

## Совместимость с новым ML

src/ml_bridge.py вызывает функции src.model и src.weather команды.
Артефакты: model_tN.joblib + booster_tN.txt + manifest_tN.json.
Поддерживаются physics, residual и direct. Для physics booster не обязателен.
Сверяются turbine_id, вариант и порядок признаков; модель, обученная на ещё
недоступных к issue_time метках, отклоняется.
metadata_tN.json и старый residual_tN.txt для team-режима не используются.

Внешний интерфейс агента требует время с offset и нормализует его в UTC.
Внутри ML — naive UTC. SCADA — фиксированный UTC+6, а не Asia/Almaty:
23:00 SCADA = 17:00Z, первая цель — 18:00Z (= 00:00 следующего дня SCADA).
История берётся через оригинальный load_turbine_hourly, сохраняющий пропуски
и требование минимум четырёх измерений на час.

Агент загружает Previous Runs, выбирает snapshot и передаёт ту же таблицу
запусков модели. Для соседних признаков сохраняются часы +0..49, выход +1..48.
Доступность определяется командным правилом run_day*24 >= lead_hours+12.
Задержка 12 часов — историческое допущение ML-команды, не доказанное время
публикации каждого выпуска. В старом metadata кэша может стоять 8: JSON агента
отдельно фиксирует реально применённую текущую константу 12, сохраняя исходный
provenance без изменения. Подробнее: [ML-описание](ml.md).

В новом ML residual_pred = prediction - level1, включая direct-модель.
Проверка суммы остаётся действительной. Отклонение от level1 > 0.3 —
предупреждение; при wind=0 и prediction>0.05 — ошибка.
Диапазон, NaN/inf, 48 последовательных часов и совпадение погоды проверяются
до сохранения.

Загрузка LightGBM нормализует CRLF через read_text/model_str: Git на Windows
иначе повреждает интерпретацию byte offsets внутри текстовой модели.
.gitattributes дополнительно сохраняет LF для booster_t*.txt.

## Опциональный generic-режим

Старые локальные адаптеры сохранены отдельно: archived_weather.py и
artifact_adapter.py. Они используются при model_backend="generic" или
CLI --model-backend generic --weather path.json. Их JSON содержит source,
issued_at, available_at, latitude, longitude, wind_unit и hourly с time,
wind_fc, temperature (48 часов; допускаются 50 с границами).
Для generic-модели нужен metadata_tN.json с порядком признаков и
residual.format ("scalar" либо "lightgbm"). Это не формат новых командных моделей.
Подстановка model_runner/weather_loader остаётся доступной для тестов.

## Проверки

```powershell
$env:OMP_NUM_THREADS = '1'
python -m unittest discover -s tests -v
python scripts/check_replay.py
python scripts/check_agent_replay.py
python -m compileall -q src tests scripts train.py predict.py
```

check_agent_replay выполняет 28 запусков для каждой из двух турбин,
проверяет 2 688 значений по сохранённому ML replay и JSON-метаданные.
Результаты записываются во временную папку, командные CSV не перезаписываются.
Совпадение с ML replay доказывает совместимость, но не MAE на скрытом
февральском факте. Метрики ML находятся в manifests и docs/ml.md.

npm/Capacitor/Electron здесь не применяются: репозиторий содержит Python.
