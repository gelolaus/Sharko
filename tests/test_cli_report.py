import json
from pathlib import Path

import pytest

from sharko import cli
from sharko.cli import main
from tests.frames import ARGS_APPROVED, ARGS_LOW_CIBIL_LONG_TERM

SYNTH_EXAMPLE = ["--example", "--loan-amount", "5000000", "--loan-term", "12"]
ANSWERS = ["1", "2", "2", "4100000", "400", "2700000", "2200000", "8800000",
           "3300000", "5000000", "12"]


@pytest.fixture
def browser_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url, *a, **k: calls.append(url) or True)
    return calls


def check(trained, capsys, *extra):
    code = main(["check", "--model", str(trained.model), "--history", "h.jsonl", *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def reports():
    folder = Path("Reports")
    return sorted(p.name for p in folder.glob("*.html")) if folder.exists() else []


def fake_input(monkeypatch, answers):
    asked = []
    queue = iter(answers)
    monkeypatch.setattr(
        "builtins.input", lambda prompt="": (asked.append(prompt), next(queue))[1]
    )
    return asked


def test_example_writes_a_report_with_no_extra_flags_and_says_where(trained, capsys, browser_calls):
    code, out, _ = check(trained, capsys, *SYNTH_EXAMPLE)
    assert code == 0 and reports() == ["Example_Applicant_Results.html"]
    assert "REPORT SAVED" in out and "Example_Applicant_Results.html" in out
    assert str(Path("Reports").resolve()) in out
    assert "--open" in out and browser_calls == []
    lines = [l for l in out.splitlines() if str(Path.cwd()) not in l]
    assert out.isascii() and max(len(l) for l in lines) <= 72


def test_names_from_flags_name_the_file_and_a_repeat_never_overwrites(trained, capsys):
    argv = ["--no-prompt", "--first-name", "Ana Maria", "--last-name", "Cruz",
            *ARGS_LOW_CIBIL_LONG_TERM]
    check(trained, capsys, *argv)
    check(trained, capsys, *argv)
    assert reports() == ["Ana_Maria_Cruz_Results.html", "Ana_Maria_Cruz_Results_2.html"]


def test_no_prompt_without_a_name_still_writes_a_report(trained, capsys):
    code, _, _ = check(trained, capsys, "--no-prompt", *ARGS_LOW_CIBIL_LONG_TERM)
    assert code == 0 and reports() == ["Unnamed_Applicant_Results.html"]


def test_reports_dir_flag_sets_the_folder(trained, capsys, tmp_path):
    check(trained, capsys, "--no-prompt", "--reports-dir", str(tmp_path / "Mine"),
          *ARGS_LOW_CIBIL_LONG_TERM)
    assert (tmp_path / "Mine" / "Unnamed_Applicant_Results.html").is_file()


def test_report_holds_the_answers_and_the_process(trained, capsys):
    check(trained, capsys, "--no-prompt", "--first-name", "Ana", "--last-name", "Cruz",
          *ARGS_LOW_CIBIL_LONG_TERM)
    html = Path("Reports", "Ana_Cruz_Results.html").read_text(encoding="utf-8")
    for text in ("Sharko report for Ana Cruz", "What you submitted", "5,000,000",
                 "How Sharko found this", "Try by try", "APPROVED"):
        assert text in html


def test_approved_application_report(trained, capsys):
    check(trained, capsys, "--no-prompt", *ARGS_APPROVED)
    html = Path("Reports", "Unnamed_Applicant_Results.html").read_text(encoding="utf-8")
    assert "Approved as submitted" in html and "Try by try" not in html


def test_names_are_not_written_to_the_history_log(trained, capsys):
    check(trained, capsys, "--no-prompt", "--first-name", "Ana", "--last-name", "Cruz",
          *ARGS_LOW_CIBIL_LONG_TERM)
    text = Path("h.jsonl").read_text(encoding="utf-8")
    assert "Ana" not in text and "Cruz" not in text and "name" not in json.loads(text)


def test_open_flag_opens_the_report_once_without_asking(
    trained, capsys, browser_calls, monkeypatch
):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask"))
    code, out, _ = check(trained, capsys, "--no-prompt", "--open", *ARGS_LOW_CIBIL_LONG_TERM)
    assert code == 0 and len(browser_calls) == 1
    assert browser_calls[0].startswith("file:") and browser_calls[0].endswith("Unnamed_Applicant_Results.html")


@pytest.mark.parametrize("reply, opened", [("y", 1), ("yes", 1), ("n", 0), ("", 0)])
def test_interactive_check_asks_name_questions_and_whether_to_open(
    trained, capsys, browser_calls, monkeypatch, reply, opened
):
    asked = fake_input(monkeypatch, ["Ana", "Cruz", *ANSWERS, reply])
    code, out, _ = check(trained, capsys)
    assert code == 0 and reports() == ["Ana_Cruz_Results.html"]
    assert "First name" in asked[0] and "Last name" in asked[1]
    assert "Open it in your browser now?" in asked[-1]
    assert "REPORT SAVED" in out and len(browser_calls) == opened


def test_invalid_name_flag_exits_2_without_a_report(trained, capsys):
    code, _, err = check(trained, capsys, "--no-prompt", "--first-name", "R2D2",
                         *ARGS_LOW_CIBIL_LONG_TERM)
    assert code == 2 and "First name" in err and reports() == []


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_cancel_at_the_name_prompt_exits_cleanly(trained, capsys, monkeypatch, error):
    def cancel(*_):
        raise error

    monkeypatch.setattr("builtins.input", cancel)
    code, _, err = check(trained, capsys)
    assert code == 130 and "Cancelled" in err and reports() == [] and not Path("h.jsonl").exists()


def test_report_command_writes_the_charts_page_into_reports(
    trained_experiment, capsys, browser_calls
):
    code = main(["report", "--out", str(trained_experiment.out)])
    out = capsys.readouterr().out
    assert code == 0 and "Experiment_Report.html" in out
    assert reports() == ["Experiment_Report.html"] and browser_calls == []
    main(["report", "--open", "--out", str(trained_experiment.out)])
    assert len(browser_calls) == 1 and browser_calls[0].endswith("Experiment_Report.html")
