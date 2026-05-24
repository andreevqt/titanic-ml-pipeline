from pathlib import Path
from src.pipeline import run_pipeline
from src.config import RAW_DATA_PATH


def test_pipeline_runs_end_to_end(tmp_path):
    run_pipeline(
        raw_path=RAW_DATA_PATH,
        processed_path=str(tmp_path / "clean.csv"),
        train_path=str(tmp_path / "train.csv"),
        test_path=str(tmp_path / "test.csv"),
        reports_dir=str(tmp_path / "reports"),
        tracking_uri=str(tmp_path / "mlruns"),
        time_budget=15,
    )

    assert (tmp_path / "clean.csv").exists()
    assert (tmp_path / "reports" / "confusion_matrix.png").exists()
    assert (tmp_path / "reports" / "roc_curve.png").exists()
    assert (tmp_path / "reports" / "drift_report.html").exists()
