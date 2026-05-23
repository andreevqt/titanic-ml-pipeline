import pytest
import pandas as pd
from src.etl import extract, transform, load
from src.config import RAW_DATA_PATH


@pytest.fixture
def raw_df():
    return extract(RAW_DATA_PATH)


@pytest.fixture
def transformed_df(raw_df):
    return transform(raw_df)


def test_no_missing_values_after_transform(transformed_df):
    assert transformed_df.isnull().sum().sum() == 0


def test_expected_columns_present(transformed_df):
    expected = {
        "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
        "Fare", "FamilySize", "IsAlone",
        "Embarked_C", "Embarked_Q", "Embarked_S",
    }
    assert expected.issubset(set(transformed_df.columns))


def test_family_size_and_is_alone(raw_df):
    df = transform(raw_df)
    solo_mask = (raw_df["SibSp"] == 0) & (raw_df["Parch"] == 0)
    assert (df.loc[solo_mask, "IsAlone"] == 1).all()
    assert (df.loc[~solo_mask, "IsAlone"] == 0).all()


def test_train_test_split_sizes(tmp_path):
    raw = extract(RAW_DATA_PATH)
    df = transform(raw)
    train_df, test_df = load(
        df,
        str(tmp_path / "clean.csv"),
        str(tmp_path / "train.csv"),
        str(tmp_path / "test.csv"),
    )
    total = len(train_df) + len(test_df)
    assert total == len(df)
    assert abs(len(test_df) / total - 0.2) < 0.05
