import pytest

from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.render import format_check_result
from sharko.search import SearchResult, Strategy
from tests.frames import ALL_NO_FLIP, APP, FLIP_RESULT, METRICS, SPACE, no_flip

TECH = "TECHNICAL DETAILS"
JARGON = (
    "normalized",
    "configuration",
    "baseline",
    "roc",
    "flip",
    "search space",
    "precision",
    "recall",
    "f1",
)
MIXED = {
    Strategy.TERM_ONLY: FLIP_RESULT,
    Strategy.AMOUNT_ONLY: no_flip(Strategy.AMOUNT_ONLY, 10),
    Strategy.COMBINED: no_flip(Strategy.COMBINED, 90),
}
LARGER_LOAN = SearchResult(
    Strategy.AMOUNT_ONLY, 500_000, 10, 0.1, True, 3, 600_000, 10, 0.1, 100_000, 20.0, 0, 0.9, 0.8
)


def render(results, score=0.1):
    return format_check_result(APP, score, results, METRICS, SPACE)


def plain_part(text):
    return text.split(TECH)[0]


def test_rejected_result_shows_plain_sections_in_order():
    text = render(MIXED)
    headings = [
        "RESULT: Not approved as submitted",
        "SMALLEST CHANGE THAT WOULD BE APPROVED",
        "HOW SHARKO FOUND THIS",
        "THE 3 WAYS COMPARED",
        TECH,
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "10.0%" in text and "50.0%" in text


def test_flip_shows_original_modified_score_and_distance():
    text = render(MIXED)
    assert "Loan term" in text and "10 -> 4 years" in text
    assert "10.0% -> 90.0%" in text
    assert "Term-only: amount 500,000 -> 500,000, term 10 -> 4 years" in text
    assert "distance 0.3333, score 0.100 -> 0.900, configurations evaluated: 5" in text
    assert "Closest by distance: Term-only" in text


def test_comparison_table_marks_closest_and_lists_tries():
    table = render(MIXED).split("THE 3 WAYS COMPARED")[1].split(TECH)[0]
    assert "Term only" in table and "<- closest" in table
    assert "10" in table and "90" in table


def test_no_flip_message_is_not_phrased_as_impossible():
    text = render(ALL_NO_FLIP)
    assert "NO APPROVED OPTION FOUND" in text
    assert "configurations evaluated: 90" in text
    assert "impossible" not in text.lower() and "<- closest" not in text
    assert "SMALLEST CHANGE" not in text


def test_approved_baseline_renders_congratulation_without_strategy_lines():
    text = render(None, score=0.93)
    assert "RESULT: Approved as submitted" in text
    assert "Congratulations!" in text and "93.0%" in text
    assert "Amount-only" not in text and "THE 3 WAYS" not in text


def test_output_always_carries_disclaimer_limitations_and_metrics():
    for results in (None, ALL_NO_FLIP, MIXED):
        text = render(results, score=0.2)
        assert DISCLAIMER in text and LIMITATIONS_NOTE in text
        assert "accuracy 0.970" in text and "ROC-AUC 0.990" in text


def test_output_states_amounts_are_indian_rupees_once():
    text = render(ALL_NO_FLIP)
    assert text.count(CURRENCY_NOTE) == 1


@pytest.mark.parametrize("results, score", [(None, 0.9), (ALL_NO_FLIP, 0.1), (MIXED, 0.1)])
def test_lines_are_short_and_ascii(results, score):
    text = render(results, score)
    assert text.isascii()
    assert max(len(line) for line in text.splitlines()) <= 72


@pytest.mark.parametrize("results, score", [(None, 0.9), (ALL_NO_FLIP, 0.1), (MIXED, 0.1)])
def test_plain_part_has_no_modeling_jargon(results, score):
    text = plain_part(render(results, score)).lower()
    assert "chance" in text
    assert [word for word in JARGON if word in text] == []


def test_larger_loan_gets_a_pattern_not_advice_note():
    larger = render({**MIXED, Strategy.AMOUNT_ONLY: LARGER_LOAN})
    assert "bigger loan" in larger.lower()
    assert "bigger loan" not in render(MIXED).lower()


def test_application_fields_use_friendly_names_in_technical_part():
    technical = render(MIXED).split(TECH)[1]
    assert "Annual income (INR): 5,000,000" in technical
    assert "CIBIL credit score: 450" in technical
    assert "income_annum" not in technical
