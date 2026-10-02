from collections.abc import Callable

import numpy as np
import pandas as pd

from sharko.config import APPROVAL_THRESHOLD, REJECTED
from sharko.data import SearchSpace
from sharko.search import Scorer, Strategy, run_all_strategies

RESULT_COLUMNS: list[str] = [
    "loan_id",
    "strategy",
    "original_amount",
    "original_term",
    "baseline_score",
    "flip_found",
    "configurations_evaluated",
    "candidate_amount",
    "candidate_term",
    "normalized_distance",
    "amount_change_abs",
    "amount_change_pct",
    "term_change_abs",
    "candidate_score",
    "score_change",
]

SUMMARY_COLUMNS: list[str] = [
    "eligible",
    "flips",
    "flip_rate_pct",
    "no_flip_rate_pct",
    "mean_distance",
    "median_distance",
    "mean_amount_change_abs",
    "mean_amount_change_pct",
    "mean_term_change_abs",
    "mean_score_change",
    "mean_configs_evaluated",
    "median_configs_evaluated",
]

PAIRED_METRICS: list[str] = [
    "normalized_distance",
    "amount_change_abs",
    "term_change_abs",
    "score_change",
    "configurations_evaluated",
]

_FLIPPED_AGGREGATES: tuple[tuple[str, str, str], ...] = (
    ("mean_distance", "normalized_distance", "mean"),
    ("median_distance", "normalized_distance", "median"),
    ("mean_amount_change_abs", "amount_change_abs", "mean"),
    ("mean_amount_change_pct", "amount_change_pct", "mean"),
    ("mean_term_change_abs", "term_change_abs", "mean"),
    ("mean_score_change", "score_change", "mean"),
)


def eligible_applications(test: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    scores = pd.Series(
        np.asarray(scorer(test), dtype=float).reshape(-1),
        index=test.index,
    )
    recorded_rejected = test["loan_status"] == REJECTED
    predicted_rejected = scores < APPROVAL_THRESHOLD
    return test.loc[recorded_rejected & predicted_rejected]


def run_experiment(
    test: pd.DataFrame,
    scorer: Scorer,
    space: SearchSpace,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    eligible = eligible_applications(test, scorer)
    total = len(eligible)
    rows: list[dict] = []
    for done, (_, applicant) in enumerate(eligible.iterrows(), start=1):
        _, results = run_all_strategies(applicant.to_dict(), scorer, space)
        for strategy in Strategy:
            row = results[strategy].as_dict()
            row["loan_id"] = applicant["loan_id"]
            rows.append(row)
        if progress is not None:
            progress(done, total)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _strategy_order(results: pd.DataFrame) -> list[str]:
    present = set(results["strategy"])
    ordered = [strategy.value for strategy in Strategy if strategy.value in present]
    seen = set(ordered)
    for strategy in results["strategy"]:
        if strategy not in seen:
            ordered.append(strategy)
            seen.add(strategy)
    return ordered


def _aggregate(series: pd.Series, how: str) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if how == "median":
        return values.median()
    return values.mean()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    index = _strategy_order(results)
    for strategy in index:
        group = results.loc[results["strategy"] == strategy]
        flipped = group.loc[group["flip_found"].eq(True)]
        eligible = int(len(group))
        flips = int(len(flipped))
        if eligible == 0:
            flip_rate = float("nan")
            no_flip_rate = float("nan")
        else:
            flip_rate = (flips / eligible) * 100
            no_flip_rate = ((eligible - flips) / eligible) * 100
        record = {
            "eligible": eligible,
            "flips": flips,
            "flip_rate_pct": flip_rate,
            "no_flip_rate_pct": no_flip_rate,
            "mean_configs_evaluated": _aggregate(
                group["configurations_evaluated"], "mean"
            ),
            "median_configs_evaluated": _aggregate(
                group["configurations_evaluated"], "median"
            ),
        }
        for name, column, how in _FLIPPED_AGGREGATES:
            record[name] = _aggregate(flipped[column], how)
        records.append(record)
    summary = pd.DataFrame(records, index=index, columns=SUMMARY_COLUMNS)
    summary.index.name = "strategy"
    return summary


def paired_comparison(
    results: pd.DataFrame,
    a: Strategy,
    b: Strategy,
) -> pd.DataFrame:
    flipped = results["flip_found"].eq(True)
    side_a = results.loc[(results["strategy"] == a.value) & flipped]
    side_b = results.loc[(results["strategy"] == b.value) & flipped]
    shared = set(side_a["loan_id"]) & set(side_b["loan_id"])
    n_both = len(shared)
    rows = []
    for metric in PAIRED_METRICS:
        if n_both == 0:
            mean_a = float("nan")
            mean_b = float("nan")
        else:
            in_both_a = side_a["loan_id"].isin(shared)
            in_both_b = side_b["loan_id"].isin(shared)
            mean_a = _aggregate(side_a.loc[in_both_a, metric], "mean")
            mean_b = _aggregate(side_b.loc[in_both_b, metric], "mean")
        rows.append(
            {
                "metric": metric,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "n_both": n_both,
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean_a", "mean_b", "n_both"])
