import pandas as pd
import pytest

from pathlib import Path

from sharko.config import DEFAULT_DATA
from sharko.data import derive_search_space, load_dataset, split_dataset


def test_load_dataset_strips_whitespace(synthetic_csv):
    df = load_dataset(synthetic_csv)
    assert all(c == c.strip() for c in df.columns)
    assert set(df["education"]) <= {"Graduate", "Not Graduate"}
    assert set(df["self_employed"]) <= {"Yes", "No"}
    assert set(df["loan_status"]) == {"Approved", "Rejected"}


def test_load_dataset_keeps_negative_residential_assets(synthetic_csv):
    df = load_dataset(synthetic_csv)
    assert len(df) == 600 and (
        df["residential_assets_value"] == -100_000
    ).any()


def test_split_is_80_20_stratified_and_seeded(synthetic_df):
    train, test = split_dataset(synthetic_df)
    assert len(train) == 480 and len(test) == 120
    assert set(train.index).isdisjoint(test.index)
    ratio = lambda d: (d["loan_status"] == "Approved").mean()
    assert abs(ratio(train) - ratio(test)) < 0.02
    train2, _ = split_dataset(synthetic_df)
    assert list(train.index) == list(train2.index)


def test_search_space_from_frame():
    train = pd.DataFrame(
        {
            "loan_amount": [500_000, 700_000, 900_000],
            "loan_term": [4, 8, 12],
        }
    )
    space = derive_search_space(train)
    assert (space.amount_min, space.amount_max, space.amount_step) == (
        500_000,
        900_000,
        100_000,
    )
    assert (
        space.terms == (4, 8, 12)
        and space.amount_range == 400_000
        and space.term_range == 8
    )


def test_search_space_uses_train_partition_only(synthetic_df):
    train, test = split_dataset(synthetic_df)
    space = derive_search_space(train)
    assert space.amount_min == train["loan_amount"].min()
    assert space.amount_max == train["loan_amount"].max()
    assert space.terms == tuple(sorted(train["loan_term"].unique()))


REAL_DATA = Path(__file__).resolve().parents[1] / DEFAULT_DATA


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real dataset not present")
def test_real_dataset_shape_and_split_sizes():
    df = load_dataset(REAL_DATA)
    assert df.shape == (4269, 13)
    assert (df["loan_status"] == "Approved").sum() == 2656
    train, test = split_dataset(df)
    assert (len(train), len(test)) == (3415, 854)
