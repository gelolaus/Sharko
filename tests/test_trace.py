import pytest

from sharko.search import Strategy, run_all_strategies
from sharko.trace import build_trace
from tests.frames import APP, SPACE, by_amount, by_term, never


@pytest.mark.parametrize("scorer", [by_term, by_amount, never])
def test_trace_matches_the_real_search_for_every_strategy(scorer):
    _, results = run_all_strategies(APP, scorer, SPACE)
    trace = build_trace(APP, scorer, SPACE, results)
    assert set(trace) == set(Strategy)
    for strategy, result in results.items():
        tries = trace[strategy]
        assert len(tries) == result.configurations_evaluated
        assert [t.number for t in tries] == list(range(1, len(tries) + 1))
        distances = [t.distance for t in tries]
        assert distances == sorted(distances)
        if result.flip_found:
            last = tries[-1]
            assert last.approved
            assert (last.amount, last.term) == (result.candidate_amount, result.candidate_term)
            assert last.score == pytest.approx(result.candidate_score)
            assert last.distance == pytest.approx(result.normalized_distance)
            assert not any(t.approved for t in tries[:-1])
        else:
            assert not any(t.approved for t in tries)


def test_trace_tries_change_only_amount_and_term_as_the_strategy_allows():
    _, results = run_all_strategies(APP, by_term, SPACE)
    trace = build_trace(APP, by_term, SPACE, results)
    assert all(t.term == APP["loan_term"] for t in trace[Strategy.AMOUNT_ONLY])
    assert all(t.amount == APP["loan_amount"] for t in trace[Strategy.TERM_ONLY])
    assert all(
        t.amount != APP["loan_amount"] and t.term != APP["loan_term"]
        for t in trace[Strategy.COMBINED]
    )
