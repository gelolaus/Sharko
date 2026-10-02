import pandas as pd

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
    assert {"pair", "metric", "mean_a", "mean_b", "n_both"} <= set(pd.read_csv(tmp_path / "paired_comparison.csv").columns)
