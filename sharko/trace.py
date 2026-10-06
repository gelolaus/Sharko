from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sharko.config import APPROVAL_THRESHOLD
from sharko.data import SearchSpace
from sharko.search import (
    Scorer,
    SearchResult,
    Strategy,
    _candidate_frame,
    generate_candidates,
)


@dataclass(frozen=True)
class Try:
    """One option the search scored: its amount, term, distance and chance."""

    number: int
    amount: int
    term: int
    distance: float
    score: float
    approved: bool


def build_trace(
    applicant: Mapping[str, Any],
    scorer: Scorer,
    space: SearchSpace,
    results: Mapping[Strategy, SearchResult],
) -> dict[Strategy, list[Try]]:
    """Rebuild every option each search scored, in order, for the report.

    This reuses the search's own candidate list and frame builder and the same
    model, and stops where the search stopped. It does not change the search.
    """
    trace: dict[Strategy, list[Try]] = {}
    for strategy, result in results.items():
        candidates = generate_candidates(
            strategy, result.original_amount, result.original_term, space
        ).head(result.configurations_evaluated)
        if len(candidates) == 0:
            trace[strategy] = []
            continue
        scores = np.asarray(
            scorer(_candidate_frame(applicant, candidates)), dtype=float
        ).reshape(-1)
        trace[strategy] = [
            Try(
                number=index + 1,
                amount=int(amount),
                term=int(term),
                distance=float(distance),
                score=float(score),
                approved=bool(score >= APPROVAL_THRESHOLD),
            )
            for index, (amount, term, distance, score) in enumerate(
                zip(
                    candidates["amount"],
                    candidates["term"],
                    candidates["distance"],
                    scores,
                )
            )
        ]
    return trace


@dataclass(frozen=True)
class Margin:
    """Approval chance when only the term, or only the amount, is varied."""

    terms: list[tuple[int, float]]
    amounts: list[tuple[int, float]]
    current_amount: int
    current_term: int

    @property
    def terms_approved(self) -> int:
        return sum(score >= APPROVAL_THRESHOLD for _, score in self.terms)

    @property
    def amounts_approved(self) -> int:
        return sum(score >= APPROVAL_THRESHOLD for _, score in self.amounts)


def margin_scan(
    applicant: Mapping[str, Any],
    scorer: Scorer,
    space: SearchSpace,
) -> Margin:
    """Score every allowed term (at the submitted amount) and every allowed amount
    (at the submitted term), to show how much room an approval has."""
    amount = int(applicant["loan_amount"])
    term = int(applicant["loan_term"])
    terms = list(space.terms)
    amounts = list(
        range(space.amount_min, space.amount_max + space.amount_step, space.amount_step)
    )

    def scores_for(frame: pd.DataFrame) -> list[float]:
        scored = scorer(_candidate_frame(applicant, frame))
        return [float(value) for value in np.asarray(scored, dtype=float).reshape(-1)]

    term_scores = scores_for(pd.DataFrame({"amount": [amount] * len(terms), "term": terms}))
    amount_scores = scores_for(pd.DataFrame({"amount": amounts, "term": [term] * len(amounts)}))
    return Margin(
        terms=list(zip(terms, term_scores)),
        amounts=list(zip(amounts, amount_scores)),
        current_amount=amount,
        current_term=term,
    )
