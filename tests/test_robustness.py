import pandas as pd
import pytest

from sharko.cli import main
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.robustness import (
    FLIP_COLUMNS,
    RUN_COLUMNS,
    seed_check,
    sensitivity,
    spread,
    without_negative_assets,
)


def test_without_negative_assets_drops_only_those_rows(synthetic_df):
    kept = without_negative_assets(synthetic_df)
    assert len(synthetic_df) - len(kept) == 5
    assert (kept["residential_assets_value"] >= 0).all()


def test_sensitivity_compares_with_and_without_the_negative_records(synthetic_df):
    table = sensitivity(synthetic_df)
    assert list(table.index) == ["all records", "without negative assets"]
    assert list(table.columns) == RUN_COLUMNS
    assert table.loc["all records", "records"] - table.loc["without negative assets", "records"] == 5
    assert table[["accuracy", "f1", "roc_auc"]].ge(0).all().all()


def test_seed_check_gives_one_row_per_seed_and_a_spread(synthetic_df):
    table = seed_check(synthetic_df, [42, 43])
    assert list(table.index) == [42, 43] and list(table.columns) == RUN_COLUMNS
    stats = spread(table)
    assert list(stats.index) == ["mean", "std", "min", "max"]
    assert (stats.loc["min"] <= stats.loc["max"]).all()
    assert set(FLIP_COLUMNS) <= set(stats.columns)


def test_seed_check_is_reproducible(synthetic_df):
    first = seed_check(synthetic_df, [42])
    second = seed_check(synthetic_df, [42])
    pd.testing.assert_frame_equal(first, second)


def test_robustness_command_prints_both_checks_and_saves_csvs(synthetic_csv, tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["robustness", "--data", str(synthetic_csv), "--seeds", "2", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "without negative assets" in text and "different random splits" in text
    assert "never a promise" in text
    assert CURRENCY_NOTE in text and DISCLAIMER in text and LIMITATIONS_NOTE in text
    assert len(pd.read_csv(out / "robustness_sensitivity.csv")) == 2
    assert len(pd.read_csv(out / "robustness_seeds.csv")) == 2


@pytest.mark.parametrize("extra, expected", [
    (["--seeds", "1"], "--seeds must be 2 or more"),
    (["--data", "nope.csv"], "Dataset not found"),
])
def test_robustness_command_rejects_bad_input(synthetic_csv, capsys, extra, expected):
    argv = ["robustness", "--data", str(synthetic_csv)]
    if "--data" in extra:
        argv = ["robustness"]
    assert main(argv + extra) == 2 and expected in capsys.readouterr().err


def test_robustness_command_can_start_from_another_seed(synthetic_csv, tmp_path, capsys):
    out = tmp_path / "o"
    argv = ["robustness", "--data", str(synthetic_csv), "--seeds", "2", "--first-seed", "100", "--out", str(out)]
    assert main(argv) == 0
    assert list(pd.read_csv(out / "robustness_seeds.csv")["seed"]) == [100, 101]
    assert "100" in capsys.readouterr().out
