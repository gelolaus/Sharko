from sharko.cli import main
from sharko.history import HistoryLog
from sharko.messages import DISCLAIMER
from tests.frames import ARGS_APPROVED, ARGS_LOW_CIBIL_LONG_TERM


def test_check_rejected_application_finds_term_flip_and_logs(trained, tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    assert (
        main(
            [
                "check",
                "--model",
                str(trained.model),
                "--history",
                str(history),
                "--no-prompt",
                *ARGS_LOW_CIBIL_LONG_TERM,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Term-only: " in out and "configurations evaluated:" in out and DISCLAIMER in out
    record = HistoryLog(history).read()[0]
    assert (
        record["source"] == "interactive"
        and record["baseline_status"] == "Rejected"
        and len(record["strategies"]) == 3
    )


def test_check_approved_application_skips_search_and_logs_no_strategies(
    trained, tmp_path, capsys
):
    history = tmp_path / "h.jsonl"
    assert (
        main(
            [
                "check",
                "--model",
                str(trained.model),
                "--history",
                str(history),
                "--no-prompt",
                *ARGS_APPROVED,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Congratulations!" in out and "configurations evaluated" not in out
    record = HistoryLog(history).read()[0]
    assert record["baseline_status"] == "Approved" and record["strategies"] is None


def test_check_result_is_independent_of_existing_history(trained, tmp_path, capsys):
    def run(history):
        main(
            [
                "check",
                "--model",
                str(trained.model),
                "--history",
                str(history),
                "--no-prompt",
                *ARGS_LOW_CIBIL_LONG_TERM,
            ]
        )
        return capsys.readouterr().out

    empty = run(tmp_path / "a.jsonl")
    (tmp_path / "b.jsonl").write_text('{"junk": 1}\n', encoding="utf-8")
    seeded = run(tmp_path / "b.jsonl")
    assert empty == seeded


def test_check_invalid_value_exits_2_without_logging(trained, tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    bad = [*ARGS_LOW_CIBIL_LONG_TERM[:-2], "--loan-term", "7"]
    assert (
        main(
            [
                "check",
                "--model",
                str(trained.model),
                "--history",
                str(history),
                "--no-prompt",
                *bad,
            ]
        )
        == 2
    )
    assert "loan_term" in capsys.readouterr().err and not history.exists()


def test_check_without_model_points_to_train(tmp_path, capsys):
    assert (
        main(["check", "--model", str(tmp_path / "x.joblib"), "--no-prompt"]) == 2
        and "Run: python -m sharko train" in capsys.readouterr().err
    )


def test_history_command_lists_runs(trained, tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    main(
        [
            "check",
            "--model",
            str(trained.model),
            "--history",
            str(history),
            "--no-prompt",
            *ARGS_LOW_CIBIL_LONG_TERM,
        ]
    )
    capsys.readouterr()
    assert main(["history", "--history", str(history)]) == 0
    out = capsys.readouterr().out
    assert "Runs logged: 1" in out and "Rejected" in out


def test_history_command_on_empty_log(tmp_path, capsys):
    assert (
        main(["history", "--history", str(tmp_path / "none.jsonl")]) == 0
        and "No history yet." in capsys.readouterr().out
    )
