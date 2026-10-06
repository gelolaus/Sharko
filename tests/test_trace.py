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


from sharko.trace import margin_scan  # noqa: E402


def test_margin_scan_scores_every_allowed_term_and_amount():
    margin = margin_scan(APP, by_term, SPACE)
    assert [term for term, _ in margin.terms] == list(SPACE.terms)
    assert margin.amounts[0][0] == SPACE.amount_min and margin.amounts[-1][0] == SPACE.amount_max
    assert len(margin.amounts) == (SPACE.amount_max - SPACE.amount_min) // SPACE.amount_step + 1
    # by_term approves terms 2 and 4 only; the amount never matters to it
    assert [term for term, score in margin.terms if score >= 0.5] == [2, 4]
    assert margin.terms_approved == 2 and margin.amounts_approved == 0


def test_margin_scan_changes_only_one_thing_at_a_time():
    seen = []

    def spy(frame):
        seen.append(frame[["loan_amount", "loan_term"]].drop_duplicates())
        return by_term(frame)

    margin_scan(APP, spy, SPACE)
    terms_frame, amounts_frame = seen[0], seen[1]
    assert set(terms_frame["loan_amount"]) == {APP["loan_amount"]}
    assert set(amounts_frame["loan_term"]) == {APP["loan_term"]}
