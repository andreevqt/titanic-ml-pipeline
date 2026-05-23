import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, REPORTS_DIR


def run_monitoring(
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
    reports_dir: str = REPORTS_DIR,
) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    column_mapping = ColumnMapping(target="Survived", prediction=None)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=train_df,
        current_data=test_df,
        column_mapping=column_mapping,
    )

    path = str(Path(reports_dir) / "drift_report.html")
    report.save_html(path)
    return path
