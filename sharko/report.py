from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sharko.experiment import paired_comparison, summarize
from sharko.search import Strategy

FOOTNOTE = "Model-based what-if results; not lender decisions."

_PAIRS: tuple[tuple[Strategy, Strategy], ...] = (
    (Strategy.AMOUNT_ONLY, Strategy.TERM_ONLY),
    (Strategy.AMOUNT_ONLY, Strategy.COMBINED),
    (Strategy.TERM_ONLY, Strategy.COMBINED),
)

_AMOUNT_STRATEGIES = (Strategy.AMOUNT_ONLY.value, Strategy.COMBINED.value)


def generate_report(results: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary = summarize(results)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, lineterminator="\n", encoding="utf-8")
    written.append(summary_path)

    paired_path = out_dir / "paired_comparison.csv"
    _paired_table(results).to_csv(
        paired_path, index=False, lineterminator="\n", encoding="utf-8"
    )
    written.append(paired_path)

    written.append(
        _bar_flip_rate(summary, out_dir / "flip_rate_by_strategy.png")
    )
    written.append(
        _box_by_strategy(
            results,
            out_dir / "distance_by_strategy.png",
            column="normalized_distance",
            title="Normalized distance by strategy",
            flipped_only=True,
        )
    )
    written.append(
        _box_by_strategy(
            results,
            out_dir / "configs_evaluated_by_strategy.png",
            column="configurations_evaluated",
            title="Configurations evaluated by strategy",
            flipped_only=False,
        )
    )

    amount_flips = _amount_flips(results)
    if len(amount_flips) > 0:
        written.append(
            _scatter_amounts(amount_flips, out_dir / "amount_original_vs_flip.png")
        )
    return written


def _paired_table(results: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for left, right in _PAIRS:
        frame = paired_comparison(results, left, right).copy()
        frame.insert(0, "pair", f"{left.value}/{right.value}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _strategy_names(results: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(results["strategy"].tolist()))
    ordered = [strategy.value for strategy in Strategy if strategy.value in present]
    for name in present:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _values_for(
    results: pd.DataFrame, strategy: str, column: str, flipped_only: bool
) -> pd.Series:
    group = results.loc[results["strategy"] == strategy]
    if flipped_only:
        group = group.loc[group["flip_found"].eq(True)]
    return pd.to_numeric(group[column], errors="coerce").dropna()


def _bar_flip_rate(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [str(name) for name in summary.index]
    ax.bar(labels, summary["flip_rate_pct"].tolist())
    ax.set_xlabel("strategy")
    ax.set_ylabel("flip_rate_pct")
    return _finish(fig, ax, "Flip rate by strategy", path)


def _box_by_strategy(
    results: pd.DataFrame,
    path: Path,
    column: str,
    title: str,
    flipped_only: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    data: list[pd.Series] = []
    labels: list[str] = []
    for name in _strategy_names(results):
        values = _values_for(results, name, column, flipped_only)
        if len(values) == 0:
            continue
        labels.append(name)
        data.append(values)
    if data:
        _draw_boxes(ax, data, labels)
    ax.set_xlabel("strategy")
    ax.set_ylabel(column)
    return _finish(fig, ax, title, path)


def _draw_boxes(ax: plt.Axes, data: list[pd.Series], labels: list[str]) -> None:
    try:
        ax.boxplot(data, tick_labels=labels)
    except TypeError:
        ax.boxplot(data, labels=labels)


def _amount_flips(results: pd.DataFrame) -> pd.DataFrame:
    flipped = results["flip_found"].eq(True)
    changes_amount = results["strategy"].isin(_AMOUNT_STRATEGIES)
    return results.loc[flipped & changes_amount]


def _scatter_amounts(flips: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(flips["original_amount"], flips["candidate_amount"])
    ax.set_xlabel("original_amount")
    ax.set_ylabel("candidate_amount")
    return _finish(fig, ax, "Original vs flipped loan amount", path)


def _finish(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> Path:
    ax.set_title(title)
    fig.text(0.5, 0.01, FOOTNOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path
