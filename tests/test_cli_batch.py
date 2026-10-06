import pandas as pd
import pytest

from sharko.cli import main
from sharko.experiment import RESULT_COLUMNS
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.model import load_bundle


def test_train_saves_model_and_prints_metrics(synthetic_csv, tmp_path, capsys):
    assert main(["train", "--data", str(synthetic_csv), "--model", str(tmp_path / "m.joblib")]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "m.joblib").exists() and "ROC-AUC" in out and "random_forest" in out
    assert load_bundle(tmp_path / "m.joblib").test_metrics["accuracy"] >= 0.95


def test_evaluate_reproduces_saved_metrics(trained, capsys):          # `trained` fixture runs train via main()
    assert main(["evaluate", "--data", str(trained.csv), "--model", str(trained.model)]) == 0
    assert f"{load_bundle(trained.model).test_metrics['accuracy']:.4f}" in capsys.readouterr().out


def test_experiment_writes_results_and_disclaimer(trained, tmp_path, capsys):
    assert main(["experiment", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(tmp_path / "o")]) == 0
    df = pd.read_csv(tmp_path / "o" / "experiment_results.csv")
    assert list(df.columns) == RESULT_COLUMNS and len(df) > 0 and len(df) % 3 == 0
    out = capsys.readouterr().out
    assert DISCLAIMER in out and LIMITATIONS_NOTE in out
    assert CURRENCY_NOTE in out


def test_experiment_is_reproducible(trained, tmp_path):
    for d in ("o1", "o2"):
        main(["experiment", "--data", str(trained.csv), "--model", str(trained.model), "--out", str(tmp_path / d)])
    assert (tmp_path / "o1" / "experiment_results.csv").read_bytes() == (tmp_path / "o2" / "experiment_results.csv").read_bytes()


def test_report_writes_figures(trained_experiment, tmp_path):
    assert main(["report", "--out", str(trained_experiment.out)]) == 0
    assert (trained_experiment.out / "flip_rate_by_strategy.png").exists()


@pytest.mark.parametrize("argv_builder, expected", [
    (lambda c, t: ["evaluate", "--data", str(c), "--model", str(t / "missing.joblib")], "Run: python -m sharko train"),
    (lambda c, t: ["experiment", "--data", str(c), "--model", str(t / "missing.joblib")], "Run: python -m sharko train"),
    (lambda c, t: ["train", "--data", str(t / "nope.csv"), "--model", str(t / "m.joblib")], "Dataset not found"),
    (lambda c, t: ["report", "--out", str(t / "empty")], "Run: python -m sharko experiment"),
])
def test_missing_artifacts_exit_2_with_actionable_message(synthetic_csv, tmp_path, capsys, argv_builder, expected):
    assert main(argv_builder(synthetic_csv, tmp_path)) == 2 and expected in capsys.readouterr().err
