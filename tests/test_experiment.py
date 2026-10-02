import pytest

from sharko.experiment import (
    RESULT_COLUMNS,
    eligible_applications,
    paired_comparison,
    run_experiment,
    summarize,
)
from sharko.search import Strategy
from tests.frames import (
    NO_OVERLAP_FIXTURE,
    RESULTS_FIXTURE,
    SPACE,
    TEST4,
    exp_scorer as scorer,
)


def test_eligibility_requires_recorded_and_predicted_rejected():
    eligible = eligible_applications(TEST4, scorer)
    assert list(eligible["loan_id"]) == [1, 4]


def test_run_experiment_shape_and_columns():
    results = run_experiment(TEST4, scorer, SPACE)
    assert list(results.columns) == RESULT_COLUMNS and len(results) == 2 * 3
    assert set(results["strategy"]) == {"amount_only", "term_only", "combined"}


def test_progress_callback_reports_each_application():
    seen = []
    run_experiment(
        TEST4,
        scorer,
        SPACE,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_summarize_rates_and_flipped_only_means():
    summary = summarize(RESULTS_FIXTURE)
    assert (
        summary.loc["term_only", "flip_rate_pct"] == 50.0
        and summary.loc["term_only", "no_flip_rate_pct"] == 50.0
    )
    assert summary.loc["term_only", "mean_distance"] == pytest.approx(0.3)
    assert (
        summary.loc["term_only", "mean_configs_evaluated"]
        == RESULTS_FIXTURE.query("strategy == 'term_only'")[
            "configurations_evaluated"
        ].mean()
    )


def test_paired_comparison_uses_only_applications_both_flipped():
    paired = paired_comparison(
        RESULTS_FIXTURE, Strategy.TERM_ONLY, Strategy.COMBINED
    )
    assert (
        paired.set_index("metric").loc["normalized_distance", "n_both"] == 1
    )


def test_paired_comparison_with_no_overlap_is_nan_not_error():
    paired = paired_comparison(
        NO_OVERLAP_FIXTURE, Strategy.AMOUNT_ONLY, Strategy.TERM_ONLY
    )
    assert (paired["n_both"] == 0).all() and paired["mean_a"].isna().all()
