import os

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
FLAML_TIME_BUDGET = int(os.getenv("FLAML_TIME_BUDGET", 60))

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/Titanic-Dataset.csv")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/titanic_clean.csv")
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/processed/train.csv")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "data/processed/test.csv")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic-survival")
