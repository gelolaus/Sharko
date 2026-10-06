import pytest

from sharko.plain import chance, heading, wrap


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "0.0%"),
        (0.003, "0.3%"),
        (0.4999, "49.9%"),
        (0.5, "50.0%"),
        (0.0026, "0.3%"),
        (0.5412, "54.1%"),
        (0.49996, "49.9%"),
        (0.50004, "50.0%"),
        (1.0, "100.0%"),
    ],
)
def test_chance_never_rounds_across_the_approval_cutoff(score, expected):
    assert chance(score) == expected


def test_wrap_indents_and_respects_width():
    lines = wrap("word " * 40, indent=4)
    assert all(line.startswith("    ") and len(line) <= 72 for line in lines)
    assert len(lines) > 1


def test_heading_is_blank_line_title_and_rule():
    assert heading("THE TITLE") == ["", "THE TITLE", "-" * 72]
