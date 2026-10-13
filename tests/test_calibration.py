import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from sharko.cli import _print_metrics, main
from sharko.data import split_dataset
from sharko.model import (
    CALIBRATION_COLUMNS,
    approval_scores,
    calibration_error,
    calibration_table,
    evaluate_model,
    train_model,
)
from sharko.report import render_calibration
from tests.frames import METRICS


def test_calibration_table_groups_scores_into_tenths():
    scores = np.array([0.05, 0.05, 0.95, 0.95])
    labels = np.array([0, 0, 1, 1])
    table = calibration_table(scores, labels)
    assert list(table.columns) == CALIBRATION_COLUMNS
    assert list(table["loans"]) == [2, 2]
    assert list(table["low"]) == [0.0, 0.9]
    assert list(table["high"]) == [0.1, 1.0]
    assert list(table["mean_score"]) == pytest.approx([0.05, 0.95])
    assert list(table["approved_rate"]) == [0.0, 1.0]


def test_calibration_table_puts_one_and_the_cut_off_in_the_right_groups():
    table = calibration_table(np.array([0.5, 1.0]), np.array([1, 1]))
    assert list(table["low"]) == [0.5, 0.9]
    assert list(table["loans"]) == [1, 1]


def test_calibration_table_leaves_out_empty_groups():
    table = calibration_table(np.array([0.31, 0.32]), np.array([0, 1]))
    assert len(table) == 1
    assert table.iloc[0]["approved_rate"] == 0.5


def test_calibration_error_is_the_loan_weighted_gap():
    scores = np.array([0.05, 0.05, 0.05, 0.95])
    labels = np.array([0, 0, 1, 1])
    # first group: mean score 0.05 vs 1/3 approved (gap 0.2833, 3 loans); second: gap 0.05
    expected = (3 * abs(0.05 - 1 / 3) + 1 * 0.05) / 4
    assert calibration_error(calibration_table(scores, labels)) == pytest.approx(expected)


def test_calibration_error_of_no_loans_is_zero():
    assert calibration_error(calibration_table(np.array([]), np.array([]))) == 0.0


def test_evaluate_model_reports_the_brier_score(synthetic_df):
    train, test = split_dataset(synthetic_df)
    bundle = train_model(train)
    metrics = evaluate_model(bundle.pipeline, test)
    labels = (test["loan_status"] == "Approved").astype(int)
    expected = brier_score_loss(labels, approval_scores(bundle.pipeline, test))
    assert metrics["brier_score"] == pytest.approx(expected)
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_metrics_print_without_a_brier_score_for_older_saved_models(capsys):
    _print_metrics(METRICS)
    assert "Brier" not in capsys.readouterr().out
    _print_metrics({**METRICS, "brier_score": 0.0123})
    assert "Brier" in capsys.readouterr().out


def test_evaluate_command_shows_brier_score_and_calibration_table(trained, tmp_path, capsys):
    argv = ["evaluate", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(tmp_path)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "Brier score" in text
    assert "Do the percentages mean what they say?" in text
    assert "Average gap" in text
    assert "estimate, not a guarantee" in text
    assert all(len(line) <= 72 for line in text.splitlines())


def test_evaluate_saves_the_calibration_table_and_chart(trained, tmp_path, capsys):
    out = tmp_path / "o"
    argv = ["evaluate", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(out)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "calibration.csv" in text and "calibration.png" in text
    saved = pd.read_csv(out / "calibration.csv", encoding="utf-8")
    assert list(saved.columns) == CALIBRATION_COLUMNS
    assert (out / "calibration.png").read_bytes().startswith(b"\x89PNG")


def test_calibration_chart_is_drawn_from_the_table(tmp_path):
    table = calibration_table(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 1, 1, 1]))
    path = render_calibration(table, tmp_path / "c.png")
    assert path.read_bytes().startswith(b"\x89PNG")


def test_report_page_includes_the_calibration_chart_only_when_it_was_saved(tmp_path):
    from sharko.report import generate_report
    from tests.frames import RESULTS_FIXTURE

    plain = generate_report(RESULTS_FIXTURE, tmp_path / "a")
    assert "Do the Percentages Mean What They Say?" not in (tmp_path / "a" / "report.html").read_text(encoding="utf-8")
    assert all(path.name != "calibration.png" for path in plain)

    table = calibration_table(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 1, 1, 1]))
    (tmp_path / "b").mkdir()
    render_calibration(table, tmp_path / "b" / "calibration.png")
    generate_report(RESULTS_FIXTURE, tmp_path / "b")
    page = (tmp_path / "b" / "report.html").read_text(encoding="utf-8")
    assert "Do the Percentages Mean What They Say?" in page
    assert page.count("<img") == 5 and page.isascii()
