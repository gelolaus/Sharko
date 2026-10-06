from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

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
