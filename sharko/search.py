from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from sharko.config import APPROVAL_THRESHOLD, FEATURES, FIXED_FEATURES
from sharko.data import SearchSpace


class Strategy(str, Enum):
    AMOUNT_ONLY = "amount_only"
    TERM_ONLY = "term_only"
    COMBINED = "combined"


Scorer = Callable[[pd.DataFrame], np.ndarray]


def _distance_numerator(
    orig_amount: int,
    orig_term: int,
    cand_amount: int,
    cand_term: int,
    space: SearchSpace,
) -> int:
    return (
        abs(cand_amount - orig_amount) * space.term_range
        + abs(cand_term - orig_term) * space.amount_range
    )


def normalized_distance(
    orig_amount: int,
    orig_term: int,
    cand_amount: int,
    cand_term: int,
    space: SearchSpace,
) -> float:
    numerator = _distance_numerator(
        orig_amount, orig_term, cand_amount, cand_term, space
    )
    return numerator / (space.amount_range * space.term_range)


def _amount_grid(space: SearchSpace) -> range:
    return range(
        space.amount_min,
        space.amount_max + space.amount_step,
        space.amount_step,
    )


def _candidate_pairs(
    strategy: Strategy,
    orig_amount: int,
    orig_term: int,
    space: SearchSpace,
) -> list[tuple[int, int]]:
    amounts = _amount_grid(space)
    terms = space.terms
    if strategy == Strategy.AMOUNT_ONLY:
        return [(amount, orig_term) for amount in amounts if amount != orig_amount]
    if strategy == Strategy.TERM_ONLY:
        return [(orig_amount, term) for term in terms if term != orig_term]
    if strategy == Strategy.COMBINED:
        return [
            (amount, term)
            for amount in amounts
            for term in terms
            if amount != orig_amount and term != orig_term
        ]
    raise ValueError("unknown strategy")


def generate_candidates(
    strategy: Strategy,
    orig_amount: int,
    orig_term: int,
    space: SearchSpace,
) -> pd.DataFrame:
    ranked = sorted(
        (
            (
                _distance_numerator(orig_amount, orig_term, amount, term, space),
                amount,
                term,
            )
            for amount, term in _candidate_pairs(
                strategy, orig_amount, orig_term, space
            )
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    denominator = space.amount_range * space.term_range
    return pd.DataFrame(
        {
            "amount": [amount for _, amount, _ in ranked],
            "term": [term for _, _, term in ranked],
            "distance": [numerator / denominator for numerator, _, _ in ranked],
        }
    )


@dataclass(frozen=True)
class SearchResult:
    strategy: Strategy
    original_amount: int
    original_term: int
    baseline_score: float
    flip_found: bool
    configurations_evaluated: int
    candidate_amount: int | None
    candidate_term: int | None
    normalized_distance: float | None
    amount_change_abs: int | None
    amount_change_pct: float | None
    term_change_abs: int | None
    candidate_score: float | None
    score_change: float | None

    @property
    def search_exhausted(self) -> bool:
        return not self.flip_found

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "original_amount": self.original_amount,
            "original_term": self.original_term,
            "baseline_score": self.baseline_score,
            "flip_found": self.flip_found,
            "configurations_evaluated": self.configurations_evaluated,
            "candidate_amount": self.candidate_amount,
            "candidate_term": self.candidate_term,
            "normalized_distance": self.normalized_distance,
            "amount_change_abs": self.amount_change_abs,
            "amount_change_pct": self.amount_change_pct,
            "term_change_abs": self.term_change_abs,
            "candidate_score": self.candidate_score,
            "score_change": self.score_change,
        }


def _feature_frame(applicant: Mapping[str, Any], count: int) -> pd.DataFrame:
    if count == 0:
        return pd.DataFrame(columns=FEATURES)
    rows = []
    for _ in range(count):
        rows.append({feature: applicant[feature] for feature in FEATURES})
    return pd.DataFrame(rows, columns=FEATURES)


def _candidate_frame(
    applicant: Mapping[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if len(candidates) == 0:
        return pd.DataFrame(columns=FEATURES)
    rows = []
    for amount, term in zip(
        candidates["amount"].tolist(),
        candidates["term"].tolist(),
    ):
        row = {feature: applicant[feature] for feature in FIXED_FEATURES}
        row["loan_amount"] = int(amount)
        row["loan_term"] = int(term)
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)


def run_search(
    strategy: Strategy,
    applicant: Mapping[str, Any],
    scorer: Scorer,
    space: SearchSpace,
    baseline_score: float,
) -> SearchResult:
    original_amount = int(applicant["loan_amount"])
    original_term = int(applicant["loan_term"])
    baseline = float(baseline_score)
    candidates = generate_candidates(strategy, original_amount, original_term, space)
    scores = np.asarray(
        scorer(_candidate_frame(applicant, candidates)),
        dtype=float,
    ).reshape(-1)
    approved = scores >= APPROVAL_THRESHOLD
    if len(candidates) and bool(np.any(approved)):
        index = int(np.argmax(approved))
        candidate_amount = int(candidates.iloc[index]["amount"])
        candidate_term = int(candidates.iloc[index]["term"])
        candidate_score = float(scores[index])
        amount_delta = abs(candidate_amount - original_amount)
        return SearchResult(
            strategy=strategy,
            original_amount=original_amount,
            original_term=original_term,
            baseline_score=baseline,
            flip_found=True,
            configurations_evaluated=index + 1,
            candidate_amount=candidate_amount,
            candidate_term=candidate_term,
            normalized_distance=float(candidates.iloc[index]["distance"]),
            amount_change_abs=amount_delta,
            amount_change_pct=(amount_delta / original_amount) * 100,
            term_change_abs=abs(candidate_term - original_term),
            candidate_score=candidate_score,
            score_change=candidate_score - baseline,
        )
    return SearchResult(
        strategy=strategy,
        original_amount=original_amount,
        original_term=original_term,
        baseline_score=baseline,
        flip_found=False,
        configurations_evaluated=len(candidates),
        candidate_amount=None,
        candidate_term=None,
        normalized_distance=None,
        amount_change_abs=None,
        amount_change_pct=None,
        term_change_abs=None,
        candidate_score=None,
        score_change=None,
    )


def run_all_strategies(
    applicant: Mapping[str, Any],
    scorer: Scorer,
    space: SearchSpace,
) -> tuple[float, dict[Strategy, SearchResult]]:
    baseline_score = float(
        np.asarray(scorer(_feature_frame(applicant, 1)), dtype=float).reshape(-1)[0]
    )
    results = {
        strategy: run_search(strategy, applicant, scorer, space, baseline_score)
        for strategy in Strategy
    }
    return baseline_score, results
