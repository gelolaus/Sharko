from datetime import datetime, timezone

import pytest

from sharko.applicant_report import write_applicant_report
from sharko.intake import FIELD_TITLES, INTAKE_FIELDS
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.search import Strategy, run_all_strategies
from sharko.trace import build_trace
from tests.frames import APP, METRICS, SPACE, by_term, never

WHEN = datetime(2026, 10, 6, 9, 30, tzinfo=timezone.utc)


def flat(text):
    return " ".join(text.split())


def make(tmp_path, scorer=by_term, first="Ana Maria", last="dela Cruz", approved=False):
    path = tmp_path / "Reports" / "Ana_Maria_dela_Cruz_Results.html"
    if approved:
        score, results, trace = 0.93, None, None
    else:
        score, results = run_all_strategies(APP, scorer, SPACE)
        trace = build_trace(APP, scorer, SPACE, results)
    written = write_applicant_report(
        path, first, last, APP, score, results, trace, METRICS, SPACE, created=WHEN
    )
    assert written == path and path.stat().st_size > 0
    return path.read_text(encoding="utf-8"), results


def test_report_shows_the_name_the_time_and_every_answer(tmp_path):
    html, _ = make(tmp_path)
    assert "Ana Maria dela Cruz" in html and "2026-10-06 09:30 UTC" in html
    for name in INTAKE_FIELDS:
        assert FIELD_TITLES[name] in html
    for value in ("Graduate", "5,000,000", "450", "3,000,000", "500,000"):
        assert value in html
    assert html.count("Can change") == 2 and html.count("Fixed") == len(INTAKE_FIELDS) - 2
    assert CURRENCY_NOTE in html


def test_report_explains_the_process_in_order(tmp_path):
    html, _ = make(tmp_path)
    headings = [
        "What you submitted",
        "The result",
        "How Sharko found this",
        "The 3 ways compared",
        "Try by try",
        "Model quality",
        "Important",
        "Words used",
    ]
    positions = [html.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Kept everything else about the applicant the same" in html
    assert "10.0%" in html and "90.0%" in html and "50.0%" in html


def test_report_try_by_try_shows_first_tries_and_the_winning_try(tmp_path):
    html, results = make(tmp_path)
    block = html.split("Try by try")[1].split("Model quality")[0]
    assert "APPROVED" in block and "not yet" in block
    for strategy, result in results.items():
        if result.flip_found:
            assert f"{result.candidate_amount:,}" in block
            assert f"<td>{result.configurations_evaluated}</td>" in block
    assert "more tries" in block


def test_report_when_nothing_flips(tmp_path):
    html, _ = make(tmp_path, scorer=never)
    assert "No approved option found" in html and "tried every option" in html
    assert "APPROVED" not in html and "(closest way)" not in html
    assert "Smallest change" not in html


def test_report_for_an_approved_application_skips_the_search_sections(tmp_path):
    html, _ = make(tmp_path, approved=True)
    assert "Approved as submitted" in html and "93.0%" in html
    assert "Try by try" not in html and "<img" not in html
    for name in INTAKE_FIELDS:
        assert FIELD_TITLES[name] in html


def test_report_is_self_contained_ascii_and_accessible(tmp_path):
    html, _ = make(tmp_path)
    assert html.isascii()
    assert "http://" not in html and "https://" not in html and "<script" not in html
    assert html.count("<img") == 1 and 'alt="' in html
    assert 'name="viewport"' in html and "prefers-color-scheme: dark" in html
    assert '<html lang="en">' in html


def test_report_carries_the_disclaimer_and_limits(tmp_path):
    html, _ = make(tmp_path)
    assert flat(DISCLAIMER) in flat(html) and flat(LIMITATIONS_NOTE) in flat(html)


def test_report_escapes_the_name(tmp_path):
    html, _ = make(tmp_path, first="<script>alert(1)</script>", last="x&y")
    assert "<script>" not in html and "&lt;script&gt;" in html and "x&amp;y" in html


def test_report_creates_the_reports_folder(tmp_path):
    assert not (tmp_path / "Reports").exists()
    make(tmp_path)
    assert (tmp_path / "Reports").is_dir()


import pandas as pd  # noqa: E402

from sharko.applicant_report import ExperimentContext  # noqa: E402
from sharko.experiment import summarize  # noqa: E402
from sharko.report import render_figures  # noqa: E402
from sharko.trace import margin_scan  # noqa: E402
from tests.frames import RESULTS_FIXTURE  # noqa: E402


def experiment_context(tmp_path):
    figures = {k: p.read_bytes() for k, p in render_figures(RESULTS_FIXTURE, tmp_path / "figs").items()}
    return ExperimentContext(figures, summarize(RESULTS_FIXTURE), 4)


def make_full(tmp_path, approved=False, experiment=None):
    path = tmp_path / "Reports" / "r.html"
    if approved:
        score, results, trace = 0.93, None, None
        margin = margin_scan(APP, by_term, SPACE)
    else:
        score, results = run_all_strategies(APP, by_term, SPACE)
        trace, margin = build_trace(APP, by_term, SPACE, results), None
    write_applicant_report(path, "Ana", "Cruz", APP, score, results, trace, METRICS, SPACE,
                           created=WHEN, experiment=experiment, margin=margin)
    return path.read_text(encoding="utf-8")


def test_approved_report_shows_the_safety_margin_chart_and_plain_sentences(tmp_path):
    html = make_full(tmp_path, approved=True)
    assert "How safe is this approval?" in html and html.count("<img") == 1
    assert "would still approve 2 of 10 loan terms" in html
    assert "0 of 11 loan amounts" in html
    assert 'alt="' in html and "Try by try" not in html


def test_report_without_experiment_results_says_how_to_add_the_paper_charts(tmp_path):
    for approved in (False, True):
        html = make_full(tmp_path, approved=approved)
        assert "How this compares with the experiment" in html
        assert "python -m sharko experiment" in html


def test_report_embeds_the_paper_charts_and_a_you_vs_experiment_table(tmp_path):
    html = make_full(tmp_path, experiment=experiment_context(tmp_path))
    assert html.count("<img") == 5            # the try chart + the 4 paper charts
    for title in PAPER_TITLES.values():
        assert title in html
    block = html.split("How this compares with the experiment")[1].split("Model quality")[0]
    assert "4 loans in the experiment" in block
    for way in ("Amount only", "Term only", "Amount and term"):
        assert way in block
    assert "25.0%" in block and "50.0%" in block        # experiment flip rates from the fixture
    assert html.isascii() and "http://" not in html and "https://" not in html


def test_approved_report_also_gets_the_paper_charts_without_the_table(tmp_path):
    html = make_full(tmp_path, approved=True, experiment=experiment_context(tmp_path))
    assert html.count("<img") == 5            # margin chart + 4 paper charts
    assert "You vs the experiment" not in html


PAPER_TITLES = {
    "flip_rate": "Prediction-Flip Rate by Strategy",
    "distance": "Minimum Normalized Distance by Strategy",
    "configs": "Configurations Evaluated by Strategy",
    "scatter": "Original vs. Prediction-Flipping Loan Amount",
}
