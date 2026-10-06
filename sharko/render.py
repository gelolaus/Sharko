from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sharko.config import APPROVAL_THRESHOLD
from sharko.data import SearchSpace
from sharko.intake import FIELD_TITLES, INTAKE_FIELDS
from sharko.messages import CURRENCY_NOTE, DISCLAIMER, LIMITATIONS_NOTE
from sharko.model import predicted_status
from sharko.plain import WIDTH, chance, heading, numbered, wrap
from sharko.search import SearchResult, Strategy

# Technical names, used in the analyst section and in the history listing.
STRATEGY_LABELS = {
    Strategy.AMOUNT_ONLY: "Amount-only",
    Strategy.TERM_ONLY: "Term-only",
    Strategy.COMBINED: "Combined",
}
# Everyday names, used in the plain-language sections.
PLAIN_WAYS = {
    Strategy.AMOUNT_ONLY: "Amount only",
    Strategy.TERM_ONLY: "Term only",
    Strategy.COMBINED: "Amount and term",
}
_CHANGE_TITLES = {
    Strategy.AMOUNT_ONLY: "Change the loan amount only:",
    Strategy.TERM_ONLY: "Change the loan term only:",
    Strategy.COMBINED: "Change both the amount and the term:",
}
_CUTOFF = chance(APPROVAL_THRESHOLD)
_MONEY = {
    "income_annum",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_amount",
}
_GLOSSARY = (
    "approval score / chance: the model's estimate from 0 to 1, shown "
    "as a percent. 0.5 (50%) or more counts as approved.",
    "distance: how far an option is from the original amount and term. "
    "0 means no change; bigger means a bigger change.",
    "configurations evaluated: how many options were scored before "
    "Sharko stopped.",
    "accuracy, precision, recall, F1, ROC-AUC: how well the model "
    "predicts loans it never saw while learning. Closer to 1 is better.",
)


def format_check_result(
    applicant: Mapping[str, Any],
    baseline_score: float,
    results: Mapping[Strategy, SearchResult] | None,
    test_metrics: Mapping[str, Any] | None,
    space: SearchSpace,
) -> str:
    lines = ["", "=" * WIDTH, "SHARKO RESULT", "=" * WIDTH, CURRENCY_NOTE]
    if results is None:
        lines.extend(_approved_lines(baseline_score))
    else:
        lines.extend(_rejected_lines(baseline_score, results, space))
    lines.extend(["", "IMPORTANT", "-" * WIDTH, DISCLAIMER, LIMITATIONS_NOTE])
    lines.extend(["", "You can stop reading here. The rest is detail for analysts."])
    lines.extend(
        _technical_lines(applicant, baseline_score, results, test_metrics, space)
    )
    return "\n".join(lines)


def _approved_lines(score: float) -> list[str]:
    lines = heading("RESULT: Approved as submitted")
    lines.extend(
        wrap(
            f"Congratulations! The model gives this application a "
            f"{chance(score)} chance of approval (it needs {_CUTOFF} or "
            "more). No change to the loan is needed.",
            indent=2,
        )
    )
    lines.extend(heading("HOW THIS WAS SCORED"))
    lines.extend(
        wrap(
            "The model compares your answers with patterns in past loan "
            "decisions and gives a chance of approval. Because the answer "
            "is already approved, Sharko did not search for changes.",
            indent=2,
        )
    )
    return lines


def _rejected_lines(
    score: float,
    results: Mapping[Strategy, SearchResult],
    space: SearchSpace,
) -> list[str]:
    closest = _closest_flipped(results)
    lines = heading("RESULT: Not approved as submitted")
    lines.extend(
        wrap(
            f"The model gives this application a {chance(score)} chance of "
            f"approval. It needs {_CUTOFF} or more to be approved.",
            indent=2,
        )
    )
    if closest is not None:
        lines.extend(_change_lines(closest, score))
    else:
        lines.extend(heading("NO APPROVED OPTION FOUND"))
        lines.extend(
            wrap(
                "Sharko tried every allowed loan amount "
                f"({space.amount_min:,} to {space.amount_max:,} INR) and "
                f"term ({space.terms[0]} to {space.terms[-1]} years). None "
                f"reached a {_CUTOFF} chance. This covers only the options "
                "Sharko tried.",
                indent=2,
            )
        )
    lines.extend(_story_lines(closest))
    lines.extend(_comparison_lines(results, closest))
    return lines


def _change_lines(closest: SearchResult, score: float) -> list[str]:
    lines = heading("SMALLEST CHANGE THAT WOULD BE APPROVED")
    lines.append(f"  {_CHANGE_TITLES[closest.strategy]}")
    if closest.candidate_amount == closest.original_amount:
        amount = f"{closest.original_amount:,} INR (unchanged)"
    else:
        amount = f"{closest.original_amount:,} -> {closest.candidate_amount:,} INR"
    if closest.candidate_term == closest.original_term:
        term = f"{closest.original_term} years (unchanged)"
    else:
        term = f"{closest.original_term} -> {closest.candidate_term} years"
    lines.append(f"    Loan amount      {amount}")
    lines.append(f"    Loan term        {term}")
    lines.append(
        f"    Approval chance  {chance(score)} -> "
        f"{chance(closest.candidate_score)} (approved)"
    )
    if closest.candidate_amount > closest.original_amount:
        lines.extend(
            wrap(
                "Note: a bigger loan is not a safer loan. The model only "
                "follows patterns in past loans, so treat this as a pattern, "
                "not as advice.",
                indent=2,
            )
        )
    return lines


def _story_lines(closest: SearchResult | None) -> list[str]:
    tries = ""
    if closest is not None:
        tries = (
            f" For the closest way, that was try {closest.configurations_evaluated:,}."
        )
    steps = [
        "Kept everything else about the applicant the same. Only the loan "
        "amount and the loan term were allowed to change.",
        "Listed the allowed amount and term options, closest to the "
        "original request first.",
        "Asked the model to score each option, one by one. Sharko stops "
        f"at the first option with a {_CUTOFF} chance or more.{tries}",
        "Did this 3 ways (amount only, term only, both) and compared them.",
    ]
    return [*heading("HOW SHARKO FOUND THIS"), *numbered(steps)]


def _comparison_lines(
    results: Mapping[Strategy, SearchResult],
    closest: SearchResult | None,
) -> list[str]:
    lines = heading("THE 3 WAYS COMPARED")
    lines.append(f"  {'Way':<17}{'Result':<8}{'Options tried':>14}")
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None:
            continue
        found = "found" if result.flip_found else "none"
        row = (
            f"  {PLAIN_WAYS[strategy]:<17}{found:<8}"
            f"{result.configurations_evaluated:>14,}"
        )
        if closest is not None and result is closest:
            row += "  <- closest"
        lines.append(row)
    return lines


def _technical_lines(
    applicant: Mapping[str, Any],
    score: float,
    results: Mapping[Strategy, SearchResult] | None,
    test_metrics: Mapping[str, Any] | None,
    space: SearchSpace,
) -> list[str]:
    lines = heading("TECHNICAL DETAILS (for analysts)")
    lines.append("Application as entered:")
    lines.extend(_applicant_lines(applicant))
    lines.append(
        f"Baseline: {predicted_status(score)} (approval score {score:.3f}, "
        f"cut-off {APPROVAL_THRESHOLD:.3f})."
    )
    if results is not None:
        lines.append("Search results (all other fields stay fixed):")
        lines.extend(_strategy_lines(results))
        closest = _closest_flipped(results)
        if closest is not None:
            lines.append(
                f"Closest by distance: {STRATEGY_LABELS[closest.strategy]}"
            )
        lines.extend(
            wrap(
                f"Search space: amounts {space.amount_min:,}-"
                f"{space.amount_max:,} INR in steps of {space.amount_step:,}; "
                f"terms {', '.join(str(term) for term in space.terms)} years.",
                hang=2,
            )
        )
    if test_metrics is not None:
        lines.extend(
            [
                "Model quality on held-out test data:",
                (
                    f"  accuracy {test_metrics['accuracy']:.3f}, "
                    f"precision {test_metrics['precision']:.3f}, "
                    f"recall {test_metrics['recall']:.3f}"
                ),
                (
                    f"  F1 {test_metrics['f1']:.3f}, "
                    f"ROC-AUC {test_metrics['roc_auc']:.3f}"
                ),
            ]
        )
    lines.append("Words used:")
    glossary = _GLOSSARY if results is not None else (_GLOSSARY[0], _GLOSSARY[3])
    for entry in glossary:
        lines.extend(wrap(f"- {entry}", indent=2, hang=2))
    return lines


def _applicant_lines(applicant: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for name in INTAKE_FIELDS:
        value = applicant[name]
        text = f"{int(value):,}" if name in _MONEY else str(value)
        lines.append(f"  {FIELD_TITLES[name]}: {text}")
    return lines


def _strategy_lines(results: Mapping[Strategy, SearchResult]) -> list[str]:
    lines: list[str] = []
    for strategy in Strategy:
        result = results.get(strategy)
        if result is None:
            continue
        label = STRATEGY_LABELS[strategy]
        if result.flip_found:
            lines.append(
                f"  {label}: amount {result.original_amount:,} -> "
                f"{result.candidate_amount:,}, term {result.original_term} -> "
                f"{result.candidate_term} years"
            )
            lines.append(
                f"    distance {result.normalized_distance:.4f}, "
                f"score {result.baseline_score:.3f} -> "
                f"{result.candidate_score:.3f}, "
                f"configurations evaluated: {result.configurations_evaluated}"
            )
        else:
            lines.append(
                f"  {label}: no flip found; configurations evaluated: "
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
