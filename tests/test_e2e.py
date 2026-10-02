import pandas as pd

from sharko.cli import main
from tests.frames import ARGS_LOW_CIBIL_LONG_TERM

PNGS = (
    "flip_rate_by_strategy.png",
    "distance_by_strategy.png",
    "configs_evaluated_by_strategy.png",
)


def test_end_to_end_pipeline(synthetic_csv, tmp_path):
    model = tmp_path / "model.joblib"
    out = tmp_path / "outputs"
    history = tmp_path / "history.jsonl"
    data = str(synthetic_csv)
    model_arg = str(model)

    assert main(["train", "--data", data, "--model", model_arg]) == 0
    assert main(["evaluate", "--data", data, "--model", model_arg]) == 0
    assert (
        main(
            [
                "experiment",
                "--data",
                data,
                "--model",
                model_arg,
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert main(["report", "--out", str(out)]) == 0
    assert (
        main(
            [
                "check",
                "--model",
                model_arg,
                "--history",
                str(history),
                "--no-prompt",
                *ARGS_LOW_CIBIL_LONG_TERM,
            ]
        )
        == 0
    )
    assert main(["history", "--history", str(history)]) == 0

    summary_path = out / "summary.csv"
    assert summary_path.is_file()
    for name in PNGS:
        assert (out / name).is_file()
    lines = [
        line
        for line in history.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1

    summary = pd.read_csv(summary_path, encoding="utf-8", index_col="strategy")
    term_rate = float(summary.loc["term_only", "flip_rate_pct"])
    amount_rate = float(summary.loc["amount_only", "flip_rate_pct"])
    assert term_rate >= 80
    assert term_rate > amount_rate
