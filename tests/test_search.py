import json

import numpy as np
import pytest

from sharko.search import (
    Strategy,
    generate_candidates,
    normalized_distance,
    run_all_strategies,
)
from tests.frames import APP, SPACE, by_amount, by_term, never


def test_normalized_distance_formula():
    assert normalized_distance(500_000, 10, 600_000, 12, SPACE) == pytest.approx(
        0.1 + 2 / 18
    )


def test_amount_only_candidates_exclude_original_and_order_by_distance_then_amount():
    c = generate_candidates(Strategy.AMOUNT_ONLY, 500_000, 10, SPACE)
    assert len(c) == 10 and 500_000 not in set(c["amount"]) and set(c["term"]) == {10}
    assert list(c["amount"][:4]) == [400_000, 600_000, 300_000, 700_000]


def test_term_only_candidates_tie_break_by_term():
    c = generate_candidates(Strategy.TERM_ONLY, 500_000, 10, SPACE)
    assert len(c) == 9 and list(c["term"][:5]) == [8, 12, 6, 14, 4]


def test_combined_requires_both_to_differ_and_breaks_ties_by_amount_then_term():
    c = generate_candidates(Strategy.COMBINED, 500_000, 10, SPACE)
    assert len(c) == 90 and ((c["amount"] != 500_000) & (c["term"] != 10)).all()
    assert tuple(c.iloc[0][["amount", "term"]]) == (400_000, 8)


def test_exact_ties_are_not_split_by_float_noise():
    c = generate_candidates(Strategy.COMBINED, 500_000, 10, SPACE)
    first_four = c.iloc[:4]
    assert len(set(first_four["distance"])) == 1 and list(first_four["amount"]) == [
        400_000,
        400_000,
        600_000,
        600_000,
    ]


def test_edge_originals_give_one_sided_bounded_candidates():
    c = generate_candidates(Strategy.AMOUNT_ONLY, 100_000, 2, SPACE)
    assert len(c) == 10 and c["amount"].min() == 200_000 and c["amount"].max() == 1_100_000
    t = generate_candidates(Strategy.TERM_ONLY, 500_000, 20, SPACE)
    assert list(t["term"][:3]) == [18, 16, 14] and t["term"].max() < 20
    outside = generate_candidates(Strategy.AMOUNT_ONLY, 2_000_000, 10, SPACE)
    assert len(outside) == 11 and outside["amount"].between(100_000, 1_100_000).all()


def test_term_only_search_finds_first_approved_in_order():
    _, r = run_all_strategies(APP, by_term, SPACE)
    t = r[Strategy.TERM_ONLY]
    assert t.flip_found and (t.candidate_term, t.candidate_amount) == (4, 500_000)
    assert t.configurations_evaluated == 5 and t.normalized_distance == pytest.approx(
        6 / 18
    )
    assert (t.term_change_abs, t.amount_change_abs) == (6, 0)
    assert t.score_change == pytest.approx(0.9 - 0.1)


def test_amount_only_search_reports_magnitudes():
    _, r = run_all_strategies(APP, by_amount, SPACE)
    a = r[Strategy.AMOUNT_ONLY]
    assert (a.candidate_amount, a.configurations_evaluated) == (200_000, 5)
    assert a.amount_change_abs == 300_000 and a.amount_change_pct == pytest.approx(60.0)
    assert not r[Strategy.TERM_ONLY].flip_found


def test_exhausted_search_reports_no_flip_with_none_fields():
    _, r = run_all_strategies(APP, never, SPACE)
    assert [r[s].configurations_evaluated for s in Strategy] == [10, 9, 90]
    x = r[Strategy.COMBINED]
    assert (
        not x.flip_found
        and x.search_exhausted
        and x.candidate_amount is None
        and x.score_change is None
    )
    json.dumps(x.as_dict())


def test_scorer_called_once_per_batch():
    calls = []
    run_all_strategies(
        APP,
        lambda df: calls.append(len(df)) or np.zeros(len(df)),
        SPACE,
    )
    assert calls == [1, 10, 9, 90]
