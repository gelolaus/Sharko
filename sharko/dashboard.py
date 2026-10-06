from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.plain import plain_way

# Chart titles follow the Pre-Final Deliverable's proposed visualizations.
FIGURE_TITLES = {
    "flip_rate": "Prediction-Flip Rate by Strategy",
    "distance": "Minimum Normalized Distance by Strategy",
    "configs": "Configurations Evaluated by Strategy",
    "scatter": "Original vs. Prediction-Flipping Loan Amount",
}
_ORDER = ("flip_rate", "distance", "configs", "scatter")
_CAPTIONS = {
    "flip_rate": (
        "How often each way found a change the model would approve. A taller "
        "bar means that way worked for more of the eligible loans."
    ),
    "distance": (
        "How big the needed change was, only for loans where a change was "
        "found. 0 means no change; a lower box means a smaller change."
    ),
    "configs": (
        "How many options were tried before stopping, on a log scale. Loans "
        "where nothing was found had every option tried."
    ),
    "scatter": (
        "Each dot is one loan where the search changed the loan amount and "
        "found an approved option. A dot on the dashed line would mean the "
        "amount did not change."
    ),
}

_CSS = """
:root {
  --bg: #ffffff; --fg: #1b1f24; --muted: #57606a; --line: #d0d7de;
  --card: #f6f8fa; --accent: #0969da;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d1117; --fg: #e6edf3; --muted: #9da7b3; --line: #30363d;
    --card: #161b22; --accent: #58a6ff;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 52rem; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.7rem; margin: 0 0 8px; }
h2 { font-size: 1.25rem; margin: 32px 0 8px; }
h3 { font-size: 1.05rem; margin: 0 0 8px; }
p { margin: 8px 0; }
.muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; min-width: 30rem; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { font-weight: 600; }
td.num, th.num { text-align: right; }
figure {
  margin: 16px 0; padding: 16px; background: var(--card);
  border: 1px solid var(--line); border-radius: 8px;
}
figure img { display: block; max-width: 100%; height: auto; margin: 0 auto; background: #fff; }
figcaption { margin-top: 8px; color: var(--muted); font-size: 0.95rem; }
.notes { white-space: pre-line; }
footer { margin-top: 32px; color: var(--muted); font-size: 0.9rem; }
"""


def _percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}%"


def _whole(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.0f}"


def _table_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for name, row in summary.iterrows():
        eligible = int(row["eligible"])
        rows.append(
            {
                "way": plain_way(str(name)),
                "eligible": f"{eligible:,}",
                "found": f"{_percent(row['flip_rate_pct'])} "
                f"({int(row['flips'])} of {eligible})",
                "tries": _whole(row["median_configs_evaluated"]),
            }
        )
    return rows


def _alt_texts(summary: pd.DataFrame) -> dict[str, str]:
    rates = ", ".join(
        f"{plain_way(str(name))} {_percent(row['flip_rate_pct'])}"
        for name, row in summary.iterrows()
    )
    tries = ", ".join(
        f"{plain_way(str(name))} {_whole(row['median_configs_evaluated'])}"
        for name, row in summary.iterrows()
    )
    return {
        "flip_rate": f"Bar chart of the share of loans with a flip: {rates}.",
        "distance": (
            "Box plot of the normalized distance of the needed change, one "
            "box per way, for loans where a flip was found."
        ),
        "configs": (
            "Box plot on a log scale of the options tried per loan, one box "
            f"per way. Median options tried: {tries}."
        ),
        "scatter": (
            "Scatter plot of the original loan amount against the amount "
            "that flipped the prediction, with a dashed no-change line."
        ),
    }


def _embed(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _figure_html(key: str, path: Path, alt: str) -> str:
    return (
        "<figure>\n"
        f"<h3>{html.escape(FIGURE_TITLES[key])}</h3>\n"
        f'<img src="{_embed(path)}" alt="{html.escape(alt, quote=True)}">\n'
        f"<figcaption>{html.escape(_CAPTIONS[key])}</figcaption>\n"
        "</figure>"
    )


def write_dashboard(
    summary: pd.DataFrame,
    eligible: int,
    figures: dict[str, Path],
    path: Path,
) -> Path:
    """Write one self-contained HTML page with every figure and plain captions."""
    alts = _alt_texts(summary)
    table_rows = "\n".join(
        "<tr><td>{way}</td><td class=\"num\">{eligible}</td>"
        "<td class=\"num\">{found}</td><td class=\"num\">{tries}</td></tr>".format(
            **{key: html.escape(value) for key, value in row.items()}
        )
        for row in _table_rows(summary)
    )
    sections = "\n".join(
        _figure_html(key, figures[key], alts[key]) for key in _ORDER if key in figures
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sharko results dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>Sharko results dashboard</h1>
<p class="muted">{html.escape(CURRENCY_NOTE)}</p>
<p>Sharko looks at loans the model rejects and tests what-if changes to the
loan amount and the loan term. Here, {eligible:,} eligible loans (marked
Rejected in the data and also rejected by the model) were searched three ways:
change the amount only, the term only, or both.</p>

<h2>At a glance</h2>
<div class="scroll">
<table>
<thead><tr><th>Way</th><th class="num">Eligible loans</th><th class="num">Approved option found</th><th class="num">Median options tried</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>

<h2>Charts</h2>
{sections}

<h2>Important</h2>
<p class="notes">{html.escape(DISCLAIMER)}</p>
<p class="notes">{html.escape(LIMITATIONS_NOTE)}</p>

<footer>Made by python -m sharko report. Files saved next to this page: summary.csv, paired_comparison.csv and the chart images.</footer>
</main>
</body>
</html>
"""
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path
