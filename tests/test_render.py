from sharko.messages import DISCLAIMER, LIMITATIONS_NOTE
from sharko.render import format_check_result
from sharko.search import Strategy
from tests.frames import ALL_NO_FLIP, APP, FLIP_RESULT, METRICS, SPACE, no_flip


def test_flip_line_shows_original_modified_distance_score_and_iterations():
    text = format_check_result(
        APP,
        0.12,
        {
            Strategy.TERM_ONLY: FLIP_RESULT,
            Strategy.AMOUNT_ONLY: no_flip(Strategy.AMOUNT_ONLY, 10),
            Strategy.COMBINED: no_flip(Strategy.COMBINED, 90),
        },
        METRICS,
        SPACE,
    )
    assert (
        "Term-only: amount 500,000 -> 500,000, term 10 -> 4 years | "
        "distance 0.3333 | score 0.100 -> 0.900 | configurations evaluated: 5"
    ) in text
    assert "Closest to your request (smallest normalized distance): Term-only" in text


def test_no_flip_message_is_not_phrased_as_impossible():
    text = format_check_result(APP, 0.1, ALL_NO_FLIP, METRICS, SPACE)
    assert (
        "no prediction flip found within the search space" in text
        and "configurations evaluated: 90" in text
    )
    assert "impossible" not in text.lower() and "Closest to your request" not in text


def test_approved_baseline_renders_congratulation_without_strategy_lines():
    text = format_check_result(APP, 0.93, None, METRICS, SPACE)
    assert "Congratulations!" in text and "(approval score 0.930)" in text
    assert "Amount-only" not in text


def test_output_always_carries_disclaimer_limitations_and_metrics():
    for results in (None, ALL_NO_FLIP):
        text = format_check_result(APP, 0.2, results, METRICS, SPACE)
        assert DISCLAIMER in text and LIMITATIONS_NOTE in text
        assert "Model test metrics: accuracy" in text
