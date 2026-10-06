import pytest

from sharko import cli
from sharko.cli import main
from sharko.history import HistoryLog
from tests.frames import ARGS_LOW_CIBIL_LONG_TERM


def run(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def assert_readable(text, ignore=None):
    assert text.isascii()
    lines = [line for line in text.splitlines() if not (ignore and ignore in line)]
    assert max(len(line) for line in lines) <= 72


def test_no_command_shows_start_screen_instead_of_an_error(capsys):
    code, out, err = run([], capsys)
    assert code == 0 and err == ""
    for command in (
        "python -m sharko train",
        "python -m sharko check --example",
        "python -m sharko tutorial",
    ):
        assert command in out
    assert_readable(out)


def test_tutorial_walks_through_install_train_check_history(capsys):
    code, out, _ = run(["tutorial"], capsys)
    assert code == 0
    for text in (
        "Step 0",
        "python -m pip install -r requirements.txt",
        "Step 1",
        "python -m sharko train",
        "python -m sharko check --example",
        "python -m sharko check",
        "python -m sharko history",
        "Troubleshooting",
        "Ctrl+C",
    ):
        assert text in out
    assert_readable(out)


def test_check_example_runs_without_questions_and_logs_as_example(
    trained, tmp_path, capsys
):
    history = tmp_path / "h.jsonl"
    code, out, _ = run(
        ["check", "--model", str(trained.model), "--history", str(history),
         "--example", "--loan-amount", "5000000", "--loan-term", "12"],
        capsys,
    )
    assert code == 0 and "built-in example application" in out and "RESULT:" in out
    assert HistoryLog(history).read()[0]["source"] == "example"


def test_check_example_values_can_be_overridden(trained, tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    code, _, _ = run(
        ["check", "--model", str(trained.model), "--history", str(history),
         "--example", "--loan-amount", "5000000", "--loan-term", "12",
         "--cibil-score", "800"],
        capsys,
    )
    assert code == 0
    assert HistoryLog(history).read()[0]["applicant"]["cibil_score"] == 800


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_check_cancel_exits_cleanly_without_logging(
    trained, tmp_path, capsys, monkeypatch, error
):
    def cancel(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(cli, "collect_applicant", cancel)
    history = tmp_path / "h.jsonl"
    code, out, err = run(
        ["check", "--model", str(trained.model), "--history", str(history)], capsys
    )
    assert code == 130 and "Cancelled" in err and not history.exists()


def test_missing_flag_error_names_the_flag(trained, tmp_path, capsys):
    code, _, err = run(
        ["check", "--model", str(trained.model), "--no-prompt",
         "--history", str(tmp_path / "h.jsonl")],
        capsys,
    )
    assert code == 2 and "--education" in err


def test_train_and_evaluate_explain_the_numbers_and_next_step(trained, capsys):
    code, out, _ = run(
        ["evaluate", "--data", str(trained.csv), "--model", str(trained.model)], capsys
    )
    assert code == 0
    assert "never saw" in out and "share of loans" in out
    assert "Next step: python -m sharko check --example" in out
    assert_readable(out)


def test_experiment_prints_a_short_plain_table_and_how_to_read_it(
    trained, tmp_path, capsys
):
    code, out, _ = run(
        ["experiment", "--data", str(trained.csv), "--model", str(trained.model),
         "--out", str(tmp_path / "o")],
        capsys,
    )
    assert code == 0
    assert "Eligible loans:" in out and "How to read this table" in out
    assert "Amount only" in out and "Amount and term" in out
    assert "median_configs_evaluated" not in out
    assert "python -m sharko report" in out
    assert_readable(out, ignore=str(tmp_path))


def test_report_describes_each_file(trained_experiment, capsys):
    code, out, _ = run(["report", "--out", str(trained_experiment.out)], capsys)
    assert code == 0
    assert "flip_rate_by_strategy.png" in out and "how often" in out.lower()
    assert "summary.csv" in out


def test_history_lines_are_short_and_have_a_legend(trained, tmp_path, capsys):
    history = tmp_path / "h.jsonl"
    main(["check", "--model", str(trained.model), "--history", str(history),
          "--no-prompt", *ARGS_LOW_CIBIL_LONG_TERM])
    capsys.readouterr()
    code, out, _ = run(["history", "--history", str(history)], capsys)
    assert code == 0 and "Runs logged: 1" in out and "chance" in out
    assert "Times are UTC" in out
    assert_readable(out)


def test_check_output_is_readable(trained, tmp_path, capsys):
    _, out, _ = run(
        ["check", "--model", str(trained.model), "--history", str(tmp_path / "h.jsonl"),
         "--no-prompt", *ARGS_LOW_CIBIL_LONG_TERM],
        capsys,
    )
    assert_readable(out)


def test_example_outside_a_custom_model_grid_explains_the_override(
    trained, tmp_path, capsys
):
    code, _, err = run(
        ["check", "--model", str(trained.model), "--example",
         "--history", str(tmp_path / "h.jsonl")],
        capsys,
    )
    assert code == 2 and "built-in example" in err and "flag" in err


@pytest.fixture
def browser_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url, *a, **k: calls.append(url) or True)
    return calls


def test_report_lists_the_dashboard_and_never_opens_it_by_default(
    trained_experiment, capsys, browser_calls
):
    code, out, _ = run(["report", "--out", str(trained_experiment.out)], capsys)
    assert code == 0 and browser_calls == []
    assert "report.html" in out and "report --open" in out
    assert (trained_experiment.out / "report.html").is_file()


def test_report_open_flag_opens_the_dashboard_once(
    trained_experiment, capsys, browser_calls
):
    code, out, _ = run(["report", "--open", "--out", str(trained_experiment.out)], capsys)
    assert code == 0 and len(browser_calls) == 1
    assert browser_calls[0].startswith("file:") and browser_calls[0].endswith("report.html")
    assert "Opening" in out


def test_tutorial_explains_how_to_get_and_where_to_find_the_charts(capsys):
    _, out, _ = run(["tutorial"], capsys)
    assert "Step 6" in out and "outputs/report.html" in out
    assert out.index("python -m sharko experiment") < out.index("python -m sharko report --open")
    assert "Run experiment first" in out
    for meaning in (
        "How often each way found an approved option",
        "How big a change was needed",
        "How many options were tried",
        "Original loan amount vs the new one",
    ):
        assert meaning in out
    assert_readable(out)


def test_start_screen_points_to_the_charts(capsys):
    _, out, _ = run([], capsys)
    assert "python -m sharko report --open" in out and "charts" in out
    assert_readable(out)
