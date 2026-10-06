import textwrap

WIDTH = 72


def chance(score: float) -> str:
    """Show a 0-1 score as a percent, never rounding across the 50% cut-off."""
    score = float(score)
    value = round(score * 100, 1)
    if score < 0.5:
        value = min(value, 49.9)
    else:
        value = max(value, 50.0)
    return f"{value:.1f}%"


def wrap(text: str, indent: int = 0, hang: int = 0) -> list[str]:
    return textwrap.wrap(
        text,
        width=WIDTH,
        initial_indent=" " * indent,
        subsequent_indent=" " * (indent + hang),
    )


def heading(title: str) -> list[str]:
    return ["", title, "-" * WIDTH]


def numbered(items: list[str], indent: int = 2) -> list[str]:
    lines: list[str] = []
    for number, item in enumerate(items, start=1):
        label = f"{number}. "
        lines.extend(
            textwrap.wrap(
                item,
                width=WIDTH,
                initial_indent=" " * indent + label,
                subsequent_indent=" " * (indent + len(label)),
            )
        )
    return lines
