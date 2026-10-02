from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sharko.data import SearchSpace
from sharko.messages import DISCLAIMER, LIMITATIONS_NOTE
from sharko.model import predicted_status
from sharko.search import SearchResult, Strategy

STRATEGY_LABELS = {
    Strategy.AMOUNT_ONLY: "Amount-only",
    Strategy.TERM_ONLY: "Term-only",
    Strategy.COMBINED: "Combined",
}


def format_check_result(
    applicant: Mapping[str, Any],
    baseline_score: float,
    results: Mapping[Strategy, SearchResult] | None,
    test_metrics: Mapping[str, Any] | None,
    space: SearchSpace,
) -> str:
    status = predicted_status(baseline_score)
    lines = [
        (
            f"Original amount {int(applicant['loan_amount']):,}, "
            f"term {int(applicant['loan_term'])} years."
        ),
        f"Baseline: {status} (approval score {baseline_score:.3f}).",
    ]
    if results is None:
        lines.append(
            "Congratulations! The model predicts this application as Approved "
            f"as submitted (approval score {baseline_score:.3f})."
        )
    else:
        lines.extend(_strategy_lines(results, space))
        closest = _closest_flipped(results)
        if closest is not None:
            label = STRATEGY_LABELS[closest.strategy]
            lines.append(
                "Closest to your request (smallest normalized distance): "
                f"{label}"
            )
    if test_metrics is not None:
        lines.append(
            "Model test metrics: "
            f"accuracy {test_metrics['accuracy']:.3f}, "
            f"precision {test_metrics['precision']:.3f}, "
            f"recall {test_metrics['recall']:.3f}, "
            f"F1 {test_metrics['f1']:.3f}, "
            f"ROC-AUC {test_metrics['roc_auc']:.3f}"
        )
    lines.append(DISCLAIMER)
    lines.append(LIMITATIONS_NOTE)
    return "\n".join(lines)


def _strategy_lines(
    results: Mapping[Strategy, SearchResult],
    space: SearchSpace,
) -> list[str]:
    terms = ", ".join(str(term) for term in space.terms)
    bounds = (
        f"amounts {space.amount_min:,}-{space.amount_max:,} "
        f"in steps of {space.amount_step:,}; terms {terms}"
    )
    lines: list[str] = []
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None:
            continue
        label = STRATEGY_LABELS[strategy]
        if result.flip_found:
            lines.append(
                f"{label}: amount {result.original_amount:,} -> "
                f"{result.candidate_amount:,}, term {result.original_term} -> "
                f"{result.candidate_term} years | "
                f"distance {result.normalized_distance:.4f} | "
                f"score {result.baseline_score:.3f} -> {result.candidate_score:.3f} | "
                f"configurations evaluated: {result.configurations_evaluated}"
            )
        else:
            lines.append(
                f"{label}: no prediction flip found within the search space "
                f"({bounds}) | configurations evaluated: "
                f"{result.configurations_evaluated}"
            )
    return lines


def _closest_flipped(
    results: Mapping[Strategy, SearchResult],
) -> SearchResult | None:
    closest: SearchResult | None = None
    closest_distance: float | None = None
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None or not result.flip_found:
            continue
        distance = result.normalized_distance
        if distance is None:
            continue
        if closest is None or distance < closest_distance:
            closest = result
            closest_distance = distance
    return closest
