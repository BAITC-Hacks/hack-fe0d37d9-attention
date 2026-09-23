from pathlib import Path

import pandas as pd

from src.scada import aggregate_hourly, complete_training_hours, load_scada_file


def test_timestamp_parsing_and_hourly_coverage(tmp_path: Path) -> None:
    path = tmp_path / "scada.csv"
    rows = [
        [index, f"2025-01-01 00:{minute:02d}:00", 5.0, 0.2, -3.0]
        for index, minute in enumerate(range(0, 60, 10), start=1)
    ]
    rows += [
        [7, "2025-01-01 01:00:00", 6.0, 0.3, -2.0],
        [8, "2025-01-01 01:10:00", 6.0, 0.3, -2.0],
    ]
    columns = [
        "ID",
        "Статистическое время",
        "Средняя скорость ветра(m/s)",
        "Нормализованная активная мощность",
        "Средняя температура окружающей среды(°C)",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")

    loaded = load_scada_file(path, "T1")
    hourly = aggregate_hourly(loaded)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert hourly["coverage_count"].tolist() == [6, 2]
    assert hourly["is_complete_hour"].tolist() == [True, False]
    assert len(complete_training_hours(hourly)) == 1
