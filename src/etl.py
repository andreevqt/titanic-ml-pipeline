import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    RANDOM_SEED, TEST_SIZE,
)


def extract(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"], errors="ignore")

    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df = df.drop(columns=["SibSp", "Parch"])

    df["Sex"] = (df["Sex"] == "male").astype(int)

    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked").astype(int)
    df = pd.concat([df.drop(columns=["Embarked"]), embarked_dummies], axis=1)

    return df


def load(
    df: pd.DataFrame,
    processed_path: str = PROCESSED_DATA_PATH,
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for p in [processed_path, train_path, test_path]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scale_cols = ["Age", "Fare", "FamilySize"]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    scaler_path = Path(processed_path).parent / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    train_df = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def run_etl(
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = extract(raw_path)
    df = transform(df)
    return load(df, processed_path, train_path, test_path)
