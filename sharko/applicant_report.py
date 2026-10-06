from __future__ import annotations

import base64
import html
import io
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sharko.config import APPROVAL_THRESHOLD
from sharko.dashboard import PAGE_CSS, figure_alt_texts, figure_html
from sharko.data import SearchSpace
from sharko.intake import FIELD_TITLES, INTAKE_FIELDS
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.plain import PLAIN_WAYS, chance, search_steps
from sharko.render import CHANGE_TITLES, GLOSSARY, closest_flipped
from sharko.report import STRATEGY_COLORS
from sharko.search import SearchResult, Strategy
from sharko.trace import Margin, Try

_CUTOFF = chance(APPROVAL_THRESHOLD)
_FIGURE_ORDER = ("flip_rate", "distance", "configs", "scatter")
_SHOWN_TRIES = 5
_CAN_CHANGE = {"loan_amount", "loan_term"}
_MONEY = {
    "income_annum",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_amount",
}
_METRIC_ROWS = (
    ("Accuracy", "accuracy", "share of loans predicted correctly"),
    ("Precision", "precision", "when it says Approved, how often it is right"),
    ("Recall", "recall", "share of truly Approved loans it caught"),
    ("F1", "f1", "balance of precision and recall"),
    ("ROC-AUC", "roc_auc", "ranks Approved above Rejected (1 = perfect)"),
)
_EXTRA_CSS = """
.badge { display: inline-block; padding: 4px 14px; border-radius: 999px;
  font-weight: 600; border: 1px solid var(--line); background: var(--card); }
ol.steps li { margin: 6px 0; }
tr.skip td { color: var(--muted); font-style: italic; }
tr.win td { font-weight: 600; }
"""


@dataclass(frozen=True)
class ExperimentContext:
    """The experiment's charts and summary, so a report can show where it fits."""

    figures: dict[str, bytes]
    summary: pd.DataFrame
    eligible: int


def _esc(value: object) -> str:
    return html.escape(str(value))


def write_applicant_report(
    path: Path,
    first: str,
    last: str,
    applicant: Mapping[str, Any],
    baseline_score: float,
    results: Mapping[Strategy, SearchResult] | None,
    trace: Mapping[Strategy, list[Try]] | None,
    test_metrics: Mapping[str, Any] | None,
    space: SearchSpace,
    created: datetime | None = None,
    experiment: ExperimentContext | None = None,
    margin: Margin | None = None,
) -> Path:
    """Write one self-contained HTML report with everything submitted and the process."""
    created = (created or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = created.strftime("%Y-%m-%d %H:%M UTC")
    name = f"{first} {last}"
    closest = closest_flipped(results) if results is not None else None

    body = [
        f"<h1>Sharko report for {_esc(name)}</h1>",
        f'<p class="muted">Created {_esc(stamp)}. {_esc(CURRENCY_NOTE)}</p>',
        _submitted_section(name, applicant),
        _result_section(baseline_score, results, closest, space),
    ]
    if results is None:
        body.append(_scored_section())
        if margin is not None:
            body.append(_margin_section(margin))
    else:
        body.append(_steps_section(closest))
        body.append(_compare_section(results, closest))
        if trace:
            body.append(_trace_section(results, trace))
    body.append(_experiment_section(results, experiment))
    if test_metrics is not None:
        body.append(_metrics_section(test_metrics))
    body.append(_important_section())
    body.append(_words_section(results is not None))

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sharko report for {_esc(name)}</title>
<style>{PAGE_CSS}{_EXTRA_CSS}</style>
</head>
<body>
<main>
{chr(10).join(body)}
<footer>Made by python -m sharko check. This file stays on your computer; nothing was sent anywhere.</footer>
</main>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def _table(headers: list[str], rows: list[str], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(
        f'<th class="num">{_esc(h)}</th>' if i in numeric else f"<th>{_esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    return (
        '<div class="scroll"><table>\n'
        f"<thead><tr>{head}</tr></thead>\n<tbody>\n"
        + "\n".join(rows)
        + "\n</tbody>\n</table></div>"
    )


def _row(cells: list[str], numeric: set[int] | None = None, css: str = "") -> str:
    numeric = numeric or set()
    tds = "".join(
        f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
        for i, cell in enumerate(cells)
    )
    return f'<tr class="{css}">{tds}</tr>' if css else f"<tr>{tds}</tr>"


def _answer(name: str, value: Any) -> str:
    return f"{int(value):,}" if name in _MONEY else str(value)


def _submitted_section(name: str, applicant: Mapping[str, Any]) -> str:
    rows = [_row(["Name", _esc(name), "Used only to name this report"])]
    for field in INTAKE_FIELDS:
        rows.append(
            _row(
                [
                    _esc(FIELD_TITLES[field]),
                    _esc(_answer(field, applicant[field])),
                    "Can change" if field in _CAN_CHANGE else "Fixed",
                ]
            )
        )
    return (
        "<h2>What you submitted</h2>\n"
        "<p>These are all the answers Sharko used. Only the loan amount and "
        "loan term may change during the search. Everything else stays fixed.</p>\n"
        + _table(["Question", "Your answer", "During the search"], rows)
    )


def _result_section(
    score: float,
    results: Mapping[Strategy, SearchResult] | None,
    closest: SearchResult | None,
    space: SearchSpace,
) -> str:
    if results is None:
        return (
            "<h2>The result</h2>\n"
            '<p><span class="badge">Approved as submitted</span></p>\n'
            f"<p>Congratulations! The model gives this application a "
            f"{chance(score)} chance of approval (it needs {_CUTOFF} or more). "
            "No change to the loan is needed.</p>"
        )
    parts = [
        "<h2>The result</h2>",
        '<p><span class="badge">Not approved as submitted</span></p>',
        f"<p>The model gives this application a {chance(score)} chance of "
        f"approval. It needs {_CUTOFF} or more to be approved.</p>",
    ]
    if closest is None:
        parts.append(
            "<h3>No approved option found</h3>\n"
            "<p>Sharko tried every allowed loan amount "
            f"({space.amount_min:,} to {space.amount_max:,} INR) and term "
            f"({space.terms[0]} to {space.terms[-1]} years). None reached a "
            f"{_CUTOFF} chance. This covers only the options Sharko tried.</p>"
        )
        return "\n".join(parts)
    amount_after = (
        f"{closest.candidate_amount:,} (unchanged)"
        if closest.candidate_amount == closest.original_amount
        else f"{closest.candidate_amount:,}"
    )
    term_after = (
        f"{closest.candidate_term} (unchanged)"
        if closest.candidate_term == closest.original_term
        else f"{closest.candidate_term}"
    )
    rows = [
        _row(["Loan amount (INR)", f"{closest.original_amount:,}", amount_after], {1, 2}),
        _row(["Loan term (years)", str(closest.original_term), term_after], {1, 2}),
        _row(["Approval chance", chance(score), chance(closest.candidate_score)], {1, 2}),
    ]
    parts.append(
        "<h3>Smallest change that would be approved</h3>\n"
        f"<p>{_esc(CHANGE_TITLES[closest.strategy])}</p>\n"
        + _table(["", "As submitted", "Changed to"], rows, {1, 2})
    )
    if closest.candidate_amount > closest.original_amount:
        parts.append(
            "<p>Note: a bigger loan is not a safer loan. The model only follows "
            "patterns in past loans, so treat this as a pattern, not as advice.</p>"
        )
    return "\n".join(parts)


def _scored_section() -> str:
    return (
        "<h2>How this was scored</h2>\n"
        "<p>The model compares your answers with patterns in past loan "
        "decisions and gives a chance of approval. Because the answer is "
        "already approved, Sharko did not search for changes.</p>"
    )


def _steps_section(closest: SearchResult | None) -> str:
    tries = closest.configurations_evaluated if closest is not None else None
    items = "\n".join(f"<li>{_esc(step)}</li>" for step in search_steps(_CUTOFF, tries))
    return f'<h2>How Sharko found this</h2>\n<ol class="steps">\n{items}\n</ol>'


def _compare_section(
    results: Mapping[Strategy, SearchResult], closest: SearchResult | None
) -> str:
    rows = []
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None:
            continue
        way = _esc(PLAIN_WAYS[strategy])
        if closest is not None and result is closest:
            way += " (closest way)"
        found = result.flip_found
        rows.append(
            _row(
                [
                    way,
                    "found" if found else "none",
                    f"{result.configurations_evaluated:,}",
                    chance(result.candidate_score) if found else "-",
                    f"{result.normalized_distance:.4f}" if found else "-",
                ],
                {2, 3, 4},
            )
        )
    return (
        "<h2>The 3 ways compared</h2>\n"
        + _table(
            ["Way", "Result", "Options tried", "Approval chance after", "Distance"],
            rows,
            {2, 3, 4},
        )
        + "\n<p class=\"muted\">Distance: 0 means no change from the request; "
        "bigger means a bigger change. The closest way has the smallest distance.</p>"
    )


def _trace_section(
    results: Mapping[Strategy, SearchResult], trace: Mapping[Strategy, list[Try]]
) -> str:
    parts = [
        "<h2>Try by try</h2>",
        "<p>This is the search itself. For each way, Sharko lists the options "
        "closest to the request first, asks the model to score each one, and "
        f"stops at the first with a {_CUTOFF} chance or more. Only the first "
        f"{_SHOWN_TRIES} tries and the last try are shown here.</p>",
    ]
    for strategy in Strategy:
        tries = trace.get(strategy)
        result = results.get(strategy)
        if not tries or result is None:
            continue
        parts.append(f"<h3>{_esc(PLAIN_WAYS[strategy])}</h3>")
        if result.flip_found:
            parts.append(
                f"<p>Sharko stopped at try {len(tries):,}, the first option with "
                f"a {_CUTOFF} chance or more.</p>"
            )
        else:
            parts.append(
                f"<p>Sharko tried every option ({len(tries):,}) and none reached "
                f"a {_CUTOFF} chance.</p>"
            )
        parts.append(_trace_table(tries, result.flip_found))
    chart = _trace_chart(results, trace)
    if chart:
        parts.append(chart)
    return "\n".join(parts)


def _try_row(item: Try) -> str:
    outcome = "APPROVED - stop here" if item.approved else "not yet"
    return _row(
        [
            str(item.number),
            f"{item.amount:,}",
            str(item.term),
            f"{item.distance:.4f}",
            chance(item.score),
            outcome,
        ],
        {1, 2, 3, 4},
        css="win" if item.approved else "",
    )


def _trace_table(tries: list[Try], flip_found: bool) -> str:
    shown = tries[:_SHOWN_TRIES]
    tail = tries[-1:] if flip_found and len(tries) > _SHOWN_TRIES else []
    hidden = len(tries) - len(shown) - len(tail)
    rows = [_try_row(item) for item in shown]
    if hidden > 0:
        rows.append(
            '<tr class="skip"><td colspan="6">... '
            f"{hidden:,} more tries, none reached {_CUTOFF} ...</td></tr>"
        )
    rows.extend(_try_row(item) for item in tail)
    return _table(
        ["Try", "Loan amount (INR)", "Loan term (years)", "Distance", "Approval chance", "Outcome"],
        rows,
        {1, 2, 3, 4},
    )


def _trace_chart(
    results: Mapping[Strategy, SearchResult], trace: Mapping[Strategy, list[Try]]
) -> str:
    ways = [s for s in Strategy if trace.get(s)]
    if not ways:
        return ""
    with plt.rc_context({"font.size": 12}):
        fig, axes = plt.subplots(
            1, len(ways), figsize=(10, 3.8), sharey=True, squeeze=False
        )
        alt_parts = _draw_trace_axes(axes[0], ways, trace)
        axes[0][0].set_ylabel("Approval chance (%)")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130)
        plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    alt = "Approval chance at each try, one small chart per way. " + "; ".join(alt_parts) + "."
    return (
        "<figure>\n<h3>Approval chance at each try</h3>\n"
        f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt, quote=True)}">\n'
        f"<figcaption>The dashed line is the {_CUTOFF} needed. A star marks the "
        "first option that reached it, where Sharko stopped.</figcaption>\n</figure>"
    )


def _draw_trace_axes(axes, ways: list[Strategy], trace: Mapping[Strategy, list[Try]]) -> list[str]:
    alt_parts: list[str] = []
    for ax, strategy in zip(axes, ways):
        tries = trace[strategy]
        xs = [item.number for item in tries]
        ys = [item.score * 100 for item in tries]
        ax.plot(
            xs, ys, color=STRATEGY_COLORS[strategy.value], linewidth=1.6,
            marker="o" if len(xs) <= 40 else None, markersize=3,
        )
        ax.axhline(APPROVAL_THRESHOLD * 100, linestyle="--", color="#555555", linewidth=1)
        ax.text(0.02, APPROVAL_THRESHOLD * 100 + 2, f"{_CUTOFF} needed",
                transform=ax.get_yaxis_transform(), fontsize=10)
        winner = [item for item in tries if item.approved]
        if winner:
            hit = winner[-1]
            ax.scatter([hit.number], [hit.score * 100], marker="*", s=170,
                       color="black", zorder=5)
            ax.annotate(f"approved at try {hit.number:,}", (hit.number, hit.score * 100),
                        xytext=(-10, 10), textcoords="offset points", ha="right", fontsize=10)
            alt_parts.append(
                f"{PLAIN_WAYS[strategy]}: approved at try {hit.number:,} "
                f"({chance(hit.score)})"
            )
        else:
            alt_parts.append(f"{PLAIN_WAYS[strategy]}: no option reached {_CUTOFF}")
        ax.set_title(PLAIN_WAYS[strategy])
        ax.set_xlabel("Try number")
        ax.set_ylim(0, 100)
    return alt_parts


def _margin_section(margin: Margin) -> str:
    terms_total, amounts_total = len(margin.terms), len(margin.amounts)
    weakest_term = min(margin.terms, key=lambda item: item[1])
    weakest_amount = min(margin.amounts, key=lambda item: item[1])
    sentences = [
        f"At your loan amount ({margin.current_amount:,} INR), the model would "
        f"still approve {margin.terms_approved} of {terms_total} loan terms. "
        f"The weakest term is {weakest_term[0]} years, at a "
        f"{chance(weakest_term[1])} chance.",
        f"At your loan term ({margin.current_term} years), the model would "
        f"still approve {margin.amounts_approved} of {amounts_total} loan "
        f"amounts. The weakest amount is {weakest_amount[0]:,} INR, at a "
        f"{chance(weakest_amount[1])} chance.",
    ]
    paragraphs = "\n".join(f"<p>{_esc(text)}</p>" for text in sentences)
    return (
        "<h2>How safe is this approval?</h2>\n"
        "<p>A yes or no does not show how much room the approval has. Here the "
        "model scores every allowed loan term, then every allowed loan amount, "
        "changing only one at a time. Everything else stays the same.</p>\n"
        + paragraphs
        + "\n"
        + _margin_chart(margin)
    )


def _margin_chart(margin: Margin) -> str:
    with plt.rc_context({"font.size": 12}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        panels = (
            (left, [t for t, _ in margin.terms], [s * 100 for _, s in margin.terms],
             margin.current_term, "Loan term (years)"),
            (right, [a / 1e6 for a, _ in margin.amounts], [s * 100 for _, s in margin.amounts],
             margin.current_amount / 1e6, "Loan amount (million INR)"),
        )
        for ax, xs, ys, current, label in panels:
            ax.plot(xs, ys, color="#0072B2", linewidth=1.8,
                    marker="o" if len(xs) <= 40 else None, markersize=4)
            ax.axhline(APPROVAL_THRESHOLD * 100, linestyle="--", color="#555555", linewidth=1)
            ax.fill_between(xs, 0, APPROVAL_THRESHOLD * 100, color="#999999", alpha=0.15)
            ax.axvline(current, color="black", linewidth=1, linestyle=":")
            at_current = ys[xs.index(current)] if current in xs else None
            if at_current is not None:
                ax.scatter([current], [at_current], s=110, color="#E69F00",
                           edgecolors="black", zorder=5)
            ax.text(0.02, APPROVAL_THRESHOLD * 100 - 8, "not approved below this line",
                    transform=ax.get_yaxis_transform(), fontsize=10)
            ax.set_xlabel(label)
            ax.set_ylim(0, 100)
        left.set_ylabel("Approval chance (%)")
        left.set_title("Only the term changes")
        right.set_title("Only the amount changes")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130)
        plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    alt = (
        "Approval chance when only the loan term changes, and when only the loan "
        f"amount changes. Still approved for {margin.terms_approved} of "
        f"{len(margin.terms)} terms and {margin.amounts_approved} of "
        f"{len(margin.amounts)} amounts."
    )
    return (
        "<figure>\n<h3>Approval chance if the loan changes</h3>\n"
        f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt, quote=True)}">\n'
        "<figcaption>The orange dot is your application. The dashed line is the "
        f"{_CUTOFF} needed; the grey area below it is not approved.</figcaption>\n</figure>"
    )


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}%"


def _num(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:,.0f}"


def _experiment_section(
    results: Mapping[Strategy, SearchResult] | None,
    experiment: ExperimentContext | None,
) -> str:
    title = "<h2>How this compares with the experiment</h2>"
    if experiment is None:
        return (
            f"{title}\n<p>The experiment tests many loans at once and draws the "
            "charts from the paper. To add them to this report, run "
            "python -m sharko experiment, then run the check again.</p>"
        )
    parts = [
        title,
        f"<p>These charts come from the experiment on {experiment.eligible:,} "
        "held-out loans that were rejected. They show how the three ways of "
        "changing a loan compare across all of them, not just yours.</p>",
    ]
    if results is not None:
        parts.append(_you_vs_experiment(results, experiment))
    alts = figure_alt_texts(experiment.summary)
    for key in _FIGURE_ORDER:
        if key in experiment.figures:
            parts.append(figure_html(key, experiment.figures[key], alts[key]))
    return "\n".join(parts)


def _you_vs_experiment(
    results: Mapping[Strategy, SearchResult], experiment: ExperimentContext
) -> str:
    rows = []
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None:
            continue
        found = result.flip_found
        if strategy.value in experiment.summary.index:
            row = experiment.summary.loc[strategy.value]
            typical = [
                _pct(row["flip_rate_pct"]),
                _num(row["median_configs_evaluated"]),
                "n/a" if pd.isna(row["median_distance"]) else f"{row['median_distance']:.4f}",
            ]
        else:
            typical = ["n/a", "n/a", "n/a"]
        rows.append(
            _row(
                [
                    _esc(PLAIN_WAYS[strategy]),
                    "found" if found else "none",
                    f"{result.configurations_evaluated:,}",
                    f"{result.normalized_distance:.4f}" if found else "-",
                    *typical,
                ],
                {2, 3, 4, 5, 6},
            )
        )
    return (
        "<h3>You vs the experiment</h3>\n"
        f"<p>Compared with the {experiment.eligible:,} loans in the experiment.</p>\n"
        + _table(
            ["Way", "Your result", "Your options tried", "Your distance",
             "Experiment: approved option found", "Experiment: median options tried",
             "Experiment: median distance"],
            rows,
            {2, 3, 4, 5, 6},
        )
    )


def _metrics_section(metrics: Mapping[str, Any]) -> str:
    rows = [
        _row([label, f"{metrics[key]:.3f}", meaning], {1})
        for label, key, meaning in _METRIC_ROWS
    ]
    return (
        "<h2>Model quality</h2>\n"
        "<p>How well the model predicts loans it never saw while learning. "
        "Closer to 1 is better.</p>\n"
        + _table(["Measure", "Score", "Meaning"], rows, {1})
    )


def _important_section() -> str:
    return (
        "<h2>Important</h2>\n"
        f'<p class="notes">{_esc(DISCLAIMER)}</p>\n'
        f'<p class="notes">{_esc(LIMITATIONS_NOTE)}</p>'
    )


def _words_section(searched: bool) -> str:
    entries = GLOSSARY if searched else (GLOSSARY[0], GLOSSARY[3])
    items = "\n".join(f"<li>{_esc(entry)}</li>" for entry in entries)
    return f"<h2>Words used</h2>\n<ul>\n{items}\n</ul>"
