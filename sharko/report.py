from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from sharko.dashboard import FIGURE_TITLES, write_dashboard
from sharko.experiment import paired_comparison, summarize
from sharko.plain import plain_way
from sharko.search import Strategy

FOOTNOTE = "Model-based what-if results; not lender decisions."

_PAIRS: tuple[tuple[Strategy, Strategy], ...] = (
    (Strategy.AMOUNT_ONLY, Strategy.TERM_ONLY),
    (Strategy.AMOUNT_ONLY, Strategy.COMBINED),
    (Strategy.TERM_ONLY, Strategy.COMBINED),
)

_AMOUNT_STRATEGIES = (Strategy.AMOUNT_ONLY.value, Strategy.COMBINED.value)

# Colour-blind-safe palette (Okabe-Ito). Values are always printed too,
# so colour never carries the meaning alone.
STRATEGY_COLORS = {
    Strategy.AMOUNT_ONLY.value: "#0072B2",
    Strategy.TERM_ONLY.value: "#E69F00",
    Strategy.COMBINED.value: "#009E73",
}
_FALLBACK_COLOR = "#999999"
FIGURE_FILES = {
    "flip_rate": "flip_rate_by_strategy.png",
    "distance": "distance_by_strategy.png",
    "configs": "configs_evaluated_by_strategy.png",
    "scatter": "amount_original_vs_flip.png",
}
_FIGSIZE = (7.5, 5.0)
_DPI = 150


def generate_report(
    results: pd.DataFrame, out_dir: Path, reports_dir: Path | None = None
) -> list[Path]:
    """Write tables, charts and the dashboard page.

    The dashboard goes to `reports_dir` as Experiment_Report.html when given,
    otherwise next to the charts as report.html.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    eligible = _eligible_count(results)

    summary = summarize(results)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, lineterminator="\n", encoding="utf-8")
    written.append(summary_path)

    paired_path = out_dir / "paired_comparison.csv"
    _paired_table(results).to_csv(
        paired_path, index=False, lineterminator="\n", encoding="utf-8"
    )
    written.append(paired_path)

    figures = render_figures(results, out_dir)
    written.extend(figures.values())

    if reports_dir is None:
        page = out_dir / "report.html"
    else:
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        page = Path(reports_dir) / "Experiment_Report.html"
    written.append(write_dashboard(summary, eligible, figures, page))
    return written


def render_figures(results: pd.DataFrame, directory: Path) -> dict[str, Path]:
    """Draw the paper's charts into `directory` (no tables, no page)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    eligible = _eligible_count(results)
    summary = summarize(results)
    figures = {
        "flip_rate": _bar_flip_rate(summary, eligible, directory / FIGURE_FILES["flip_rate"]),
        "distance": _box_distance(results, eligible, directory / FIGURE_FILES["distance"]),
        "configs": _box_configs(results, eligible, directory / FIGURE_FILES["configs"]),
    }
    amount_flips = _amount_flips(results)
    if len(amount_flips) > 0:
        figures["scatter"] = _scatter_amounts(
            amount_flips, eligible, directory / FIGURE_FILES["scatter"]
        )
    return figures


def figure_bytes(
    results: pd.DataFrame, out_dir: Path, results_path: Path
) -> dict[str, bytes]:
    """PNG bytes of the paper's charts: reuse the saved ones if they are newer than
    the experiment results, otherwise draw them fresh in a temporary folder."""
    out_dir = Path(out_dir)
    saved = {
        key: out_dir / name
        for key, name in FIGURE_FILES.items()
        if (out_dir / name).is_file()
    }
    newest = Path(results_path).stat().st_mtime
    required = ("flip_rate", "distance", "configs")
    if all(key in saved and saved[key].stat().st_mtime >= newest for key in required):
        return {
            key: path.read_bytes()
            for key, path in saved.items()
            if path.stat().st_mtime >= newest
        }
    with tempfile.TemporaryDirectory() as scratch:
        figures = render_figures(results, Path(scratch))
        return {key: path.read_bytes() for key, path in figures.items()}


def _eligible_count(results: pd.DataFrame) -> int:
    return int(results["loan_id"].nunique()) if len(results) else 0


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


def _color(name: str) -> str:
    return STRATEGY_COLORS.get(name, _FALLBACK_COLOR)


def _bar_flip_rate(summary: pd.DataFrame, eligible: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    names = [str(name) for name in summary.index]
    rates = [float(rate) for rate in summary["flip_rate_pct"]]
    heights = [0.0 if pd.isna(rate) else rate for rate in rates]
    bars = ax.bar(
        [plain_way(name) for name in names],
        heights,
        color=[_color(name) for name in names],
        edgecolor="black",
    )
    for bar, rate, (_, row) in zip(bars, rates, summary.iterrows()):
        if pd.isna(rate):
            continue
        ax.annotate(
            f"{rate:.1f}%\n({int(row['flips'])} of {int(row['eligible'])})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Way of changing the loan")
    ax.set_ylabel("Eligible loans with an approved option found (%)")
    _note_sample(fig, eligible)
    return _finish(fig, ax, FIGURE_TITLES["flip_rate"], path)


def _box_distance(results: pd.DataFrame, eligible: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    names, data = [], []
    for name in _strategy_names(results):
        values = _values_for(results, name, "normalized_distance", True)
        if len(values) == 0:
            continue
        names.append(name)
        data.append(values)
    if data:
        labels = [
            f"{plain_way(name)}\n(n = {len(values)})"
            for name, values in zip(names, data)
        ]
        _draw_boxes(ax, data, labels, names)
    else:
        ax.text(0.5, 0.5, "No approved option was found", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_xlabel("Way of changing the loan (n = loans where an option was found)")
    ax.set_ylabel("Normalized distance (0 = no change)")
    _note_sample(fig, eligible)
    return _finish(fig, ax, FIGURE_TITLES["distance"], path)


def _box_configs(results: pd.DataFrame, eligible: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    names, data, labels = [], [], []
    for name in _strategy_names(results):
        values = _values_for(results, name, "configurations_evaluated", False)
        if len(values) == 0:
            continue
        group = results.loc[results["strategy"] == name]
        found = int(group["flip_found"].eq(True).sum())
        names.append(name)
        data.append(values.clip(lower=1))
        labels.append(
            f"{plain_way(name)}\nfound: {found}\nsearched all: {len(group) - found}"
        )
    if data:
        _draw_boxes(ax, data, labels, names)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ax.set_xticks([])
    ax.set_xlabel("Way of changing the loan")
    ax.set_ylabel("Options tried before stopping (log scale)")
    _note_sample(fig, eligible)
    return _finish(fig, ax, FIGURE_TITLES["configs"], path)


def _draw_boxes(
    ax: plt.Axes, data: list[pd.Series], labels: list[str], names: list[str]
) -> None:
    try:
        parts = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        parts = ax.boxplot(data, labels=labels, patch_artist=True)
    for box, name in zip(parts["boxes"], names):
        box.set_facecolor(_color(name))
        box.set_alpha(0.75)
    for median in parts["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)


def _amount_flips(results: pd.DataFrame) -> pd.DataFrame:
    flipped = results["flip_found"].eq(True)
    changes_amount = results["strategy"].isin(_AMOUNT_STRATEGIES)
    return results.loc[flipped & changes_amount]


def _scatter_amounts(flips: pd.DataFrame, eligible: int, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    markers = {Strategy.AMOUNT_ONLY.value: "o", Strategy.COMBINED.value: "s"}
    handles = []
    # Draw the many Combined dots first so the few Amount-only dots stay visible on top.
    for layer, name in enumerate(reversed(_AMOUNT_STRATEGIES), start=2):
        group = flips.loc[flips["strategy"] == name]
        if len(group) == 0:
            continue
        ax.scatter(
            group["original_amount"] / 1e6,
            group["candidate_amount"] / 1e6,
            color=_color(name),
            marker=markers[name],
            s=28 + 22 * (layer - 2),
            alpha=0.55 + 0.4 * (layer - 2),
            edgecolors="black",
            linewidths=0.4,
            zorder=layer,
        )
        handles.insert(
            0,
            Line2D([], [], color=_color(name), marker=markers[name], linestyle="",
                   markeredgecolor="black", label=plain_way(name)),
        )
    low = float(min(flips["original_amount"].min(), flips["candidate_amount"].min())) / 1e6
    high = float(max(flips["original_amount"].max(), flips["candidate_amount"].max())) / 1e6
    ax.plot([low, high], [low, high], linestyle="--", color="#555555", linewidth=1)
    handles.append(
        Line2D([], [], color="#555555", linestyle="--", label="no change in amount")
    )
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlabel("Original loan amount (million INR)")
    ax.set_ylabel("Amount in the approved option (million INR)")
    _note_sample(fig, eligible)
    return _finish(fig, ax, FIGURE_TITLES["scatter"], path)


def _note_sample(fig: plt.Figure, eligible: int) -> None:
    fig.text(0.5, 0.05, f"n = {eligible} eligible loans", ha="center", fontsize=9)


def _finish(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> Path:
    ax.set_title(title)
    fig.text(0.5, 0.01, FOOTNOTE, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path
