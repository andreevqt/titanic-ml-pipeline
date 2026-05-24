import pytest
import mlflow
from src.etl import extract, transform, load
from src.train import run_training
from src.config import RAW_DATA_PATH


@pytest.fixture(scope="module")
def data_splits(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    df = extract(RAW_DATA_PATH)
    df = transform(df)
    load(
        df,
        str(tmp / "clean.csv"),
        str(tmp / "train.csv"),
        str(tmp / "test.csv"),
    )
    return str(tmp / "train.csv"), str(tmp / "test.csv")


def test_model_beats_baseline(tmp_path, data_splits):
    train_path, test_path = data_splits
    result = run_training(
        train_path=train_path,
        test_path=test_path,
        tracking_uri=str(tmp_path / "mlruns"),
        time_budget=15,
    )
    assert result["val_roc_auc"] > 0.7


def test_mlflow_run_logged(tmp_path, data_splits):
    train_path, test_path = data_splits
    tracking_uri = str(tmp_path / "mlruns2")
    result = run_training(
        train_path=train_path,
        test_path=test_path,
        tracking_uri=tracking_uri,
        time_budget=15,
    )
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(result["run_id"])
    assert "val_roc_auc" in run.data.metrics
