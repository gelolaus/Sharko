import pandas as pd
import pytest

from sharko.cli import main
from sharko.data import derive_search_space, split_dataset
from sharko.experiment import run_experiment, summarize
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.scenarios import (
    BREAKDOWN_COLUMNS,
    PROFILE_DIMENSIONS,
    format_value,
    pick_examples,
    profile_breakdown,
)
from tests.frames import exp_scorer


@pytest.fixture(scope="module")
def run():
    from tests.synthetic import make_synthetic_df

    train, test = split_dataset(make_synthetic_df(n=900))
    results = run_experiment(test, exp_scorer, derive_search_space(train))
    return results, test


def test_breakdown_has_every_profile_band_and_strategy(run):
    results, test = run
    table = profile_breakdown(results, test)
    assert list(table.columns) == BREAKDOWN_COLUMNS
    assert set(table["dimension"]) == {title for _, title in PROFILE_DIMENSIONS}
    assert set(table["strategy"]) == {"amount_only", "term_only", "combined"}
    assert set(table["band"]) == {"lowest third", "middle third", "highest third"}


def test_each_dimension_splits_the_eligible_loans_without_loss(run):
    results, test = run
    table = profile_breakdown(results, test)
    overall = summarize(results)
    for _, title in PROFILE_DIMENSIONS:
        part = table[table["dimension"] == title].groupby("strategy")[["eligible", "flips"]].sum()
        for strategy in overall.index:
            assert part.loc[strategy, "eligible"] == overall.loc[strategy, "eligible"]
            assert part.loc[strategy, "flips"] == overall.loc[strategy, "flips"]


def test_bands_run_from_low_to_high_without_overlap(run):
    results, test = run
    table = profile_breakdown(results, test)
    cibil = table[(table["dimension"] == "CIBIL credit score") & (table["strategy"] == "combined")]
    cibil = cibil.set_index("band")
    assert cibil.loc["lowest third", "high"] <= cibil.loc["middle third", "low"]
    assert cibil.loc["middle third", "high"] <= cibil.loc["highest third", "low"]


def test_breakdown_is_empty_but_well_formed_without_results(run):
    _, test = run
    table = profile_breakdown(pd.DataFrame(columns=run[0].columns), test)
    assert table.empty and list(table.columns) == BREAKDOWN_COLUMNS


def test_examples_pick_real_eligible_loans_by_profile(run):
    results, test = run
    examples = pick_examples(results, test)
    labels = [label for label, _ in examples]
    assert labels == [
        "Lowest CIBIL score",
        "Middle CIBIL score",
        "Highest CIBIL score",
        "Largest loan for the income",
    ]
    eligible = test[test["loan_id"].isin(results["loan_id"])]
    ids = dict(examples)
    assert ids["Lowest CIBIL score"] == eligible.sort_values(["cibil_score", "loan_id"]).iloc[0]["loan_id"]
    assert ids["Highest CIBIL score"] == eligible.sort_values(["cibil_score", "loan_id"]).iloc[-1]["loan_id"]
    assert set(ids.values()) <= set(results["loan_id"])
    assert examples == pick_examples(results, test)


def test_scenarios_command_prints_examples_breakdown_and_saves_csv(trained, tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["scenarios", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Lowest CIBIL score" in text and "CIBIL credit score" in text
    assert "Amount only" in text and "Term only" in text and "Amount and term" in text
    assert CURRENCY_NOTE in text and DISCLAIMER in text and LIMITATIONS_NOTE in text
    saved = pd.read_csv(out / "scenarios.csv")
    assert list(saved.columns) == BREAKDOWN_COLUMNS and len(saved) > 0


@pytest.mark.parametrize("flag, expected", [
    ("--model", "Run: python -m sharko train"),
    ("--data", "Dataset not found"),
])
def test_scenarios_command_reports_missing_files(synthetic_csv, tmp_path, capsys, flag, expected):
    argv = ["scenarios", "--data", str(synthetic_csv), "--model", str(tmp_path / "missing.joblib")]
    if flag == "--data":
        argv = ["scenarios", "--data", str(tmp_path / "nope.csv")]
    assert main(argv) == 2 and expected in capsys.readouterr().err


def test_values_print_short_enough_to_fit_a_72_column_row():
    assert format_value("cibil_score", 417.0) == "417"
    assert format_value("income_annum", 4_100_000) == "4.1M"
    assert format_value("loan_to_income", 2.9756) == "2.98"


def test_breakdown_with_fewer_than_three_loans_uses_one_group(run):
    results, test = run
    two = results[results["loan_id"].isin(results["loan_id"].unique()[:2])]
    table = profile_breakdown(two, test)
    assert set(table["band"]) == {"lowest third"}
    assert (table["eligible"] == 2).all()


def test_scenarios_command_keeps_every_line_within_72_columns(trained, tmp_path, capsys):
    main(["scenarios", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(tmp_path / "o")])
    lines = [l for l in capsys.readouterr().out.splitlines() if not l.startswith("Saved to")]
    assert all(len(line) <= 72 for line in lines)
