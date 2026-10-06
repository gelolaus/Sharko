import pandas as pd

from sharko import report
from sharko.experiment import summarize
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.report import generate_report
from tests.frames import NO_FLIPS_FIXTURE, RESULTS_FIXTURE, TERM_ONLY_FLIPS_FIXTURE


def test_generate_report_writes_expected_files(tmp_path):
    paths = generate_report(RESULTS_FIXTURE, tmp_path)
    names = {p.name for p in paths}
    assert {"summary.csv", "paired_comparison.csv", "flip_rate_by_strategy.png",
            "distance_by_strategy.png", "configs_evaluated_by_strategy.png"} <= names
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_scatter_only_when_amount_flips_exist(tmp_path):
    names = {p.name for p in generate_report(TERM_ONLY_FLIPS_FIXTURE, tmp_path)}
    assert "amount_original_vs_flip.png" not in names


def test_report_handles_strategy_with_no_flips(tmp_path):
    generate_report(NO_FLIPS_FIXTURE, tmp_path)      # must not raise on empty distance series


def test_paired_comparison_csv_has_pair_column(tmp_path):
    generate_report(RESULTS_FIXTURE, tmp_path)
    assert {"pair", "metric", "mean_a", "mean_b", "n_both"} <= set(pd.read_csv(tmp_path / "paired_comparison.csv", encoding="utf-8").columns)


PAPER_TITLES = {
    "flip_rate": "Prediction-Flip Rate by Strategy",
    "distance": "Minimum Normalized Distance by Strategy",
    "configs": "Configurations Evaluated by Strategy",
    "scatter": "Original vs. Prediction-Flipping Loan Amount",
}


def flat(text):
    return " ".join(text.split())


def dashboard(results, tmp_path):
    paths = generate_report(results, tmp_path)
    page = tmp_path / "report.html"
    assert page in paths
    return page.read_text(encoding="utf-8")


def test_dashboard_is_self_contained_and_ascii(tmp_path):
    html = dashboard(RESULTS_FIXTURE, tmp_path)
    assert html.isascii()
    assert html.count("<img") == 4 and html.count("data:image/png;base64,") == 4
    assert "http://" not in html and "https://" not in html and "<script" not in html
    assert html.count('alt="') >= 4
    assert 'name="viewport"' in html and "prefers-color-scheme: dark" in html


def test_dashboard_shows_paper_titles_counts_and_notes(tmp_path):
    html = dashboard(RESULTS_FIXTURE, tmp_path)
    for title in PAPER_TITLES.values():
        assert title in html
    assert "4 eligible loans" in html
    assert flat(DISCLAIMER) in flat(html) and flat(LIMITATIONS_NOTE) in flat(html)
    assert CURRENCY_NOTE in html


def test_dashboard_table_has_plain_names_and_rates(tmp_path):
    html = dashboard(RESULTS_FIXTURE, tmp_path)
    for name in ("Amount only", "Term only", "Amount and term"):
        assert name in html
    assert "25.0%" in html and "50.0%" in html


def test_dashboard_skips_scatter_when_no_amount_flips(tmp_path):
    html = dashboard(TERM_ONLY_FLIPS_FIXTURE, tmp_path)
    assert PAPER_TITLES["scatter"] not in html and html.count("<img") == 3


def test_dashboard_renders_when_nothing_flips(tmp_path):
    html = dashboard(NO_FLIPS_FIXTURE, tmp_path)
    assert "0.0%" in html and html.count("<img") == 3


def test_charts_use_the_paper_titles(tmp_path, monkeypatch):
    seen = []
    original = report._finish

    def record(fig, ax, title, path):
        seen.append(title)
        return original(fig, ax, title, path)

    monkeypatch.setattr(report, "_finish", record)
    generate_report(RESULTS_FIXTURE, tmp_path)
    assert set(seen) == set(PAPER_TITLES.values())


def test_summary_csv_is_unchanged_by_the_dashboard(tmp_path):
    generate_report(RESULTS_FIXTURE, tmp_path)
    expected = summarize(RESULTS_FIXTURE).to_csv(lineterminator="\n")
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8") == expected
