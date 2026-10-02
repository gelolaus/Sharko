import numpy as np
import pandas as pd

from sharko.data import SearchSpace
from sharko.search import SearchResult, Strategy

SPACE = SearchSpace(100_000, 1_100_000, 100_000, tuple(range(2, 21, 2)))
FIXED = {
    "education": "Graduate",
    "self_employed": "No",
    "no_of_dependents": 2,
    "income_annum": 5_000_000,
    "cibil_score": 450,
    "residential_assets_value": 2_000_000,
    "commercial_assets_value": 1_000_000,
    "luxury_assets_value": 3_000_000,
    "bank_asset_value": 1_500_000,
}
APP = {**FIXED, "loan_amount": 500_000, "loan_term": 10}
by_term = lambda df: np.where(df["loan_term"] <= 4, 0.9, 0.1)
by_amount = lambda df: np.where(df["loan_amount"] <= 200_000, 0.9, 0.1)
never = lambda df: np.zeros(len(df))

exp_scorer = lambda df: np.where(
    (df["cibil_score"] >= 550) | (df["loan_term"] <= 4), 0.9, 0.1
)

_RESULT_COLUMNS = [
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


def _row(loan_id, status, cibil, amount, term):
    return {
        "loan_id": loan_id,
        **FIXED,
        "cibil_score": cibil,
        "loan_amount": amount,
        "loan_term": term,
        "loan_status": status,
    }


TEST4 = pd.DataFrame(
    [
        _row(1, "Rejected", 450, 500_000, 10),
        _row(2, "Rejected", 700, 500_000, 10),
        _row(3, "Approved", 450, 500_000, 10),
        _row(4, "Rejected", 450, 600_000, 12),
    ]
)


def make_results_frame(rows):
    """rows: list of (loan_id, strategy, flip, configs, distance). Returns a frame with RESULT_COLUMNS.
    Every original is amount 500_000, term 10, baseline_score 0.1. Flipped rows: candidate_score 0.9,
    score_change 0.8, candidate_amount 400_000 for amount_only/combined else 500_000, candidate_term 10 for
    amount_only else 6, amount_change_abs = |candidate - 500_000|, amount_change_pct = that / 500_000 * 100,
    term_change_abs = |candidate_term - 10|. Non-flipped rows: every candidate_/change/score_change/distance field is None."""
    records = []
    for loan_id, strategy, flip, configs, distance in rows:
        record = {
            "loan_id": loan_id,
            "strategy": strategy,
            "original_amount": 500_000,
            "original_term": 10,
            "baseline_score": 0.1,
            "flip_found": flip,
            "configurations_evaluated": configs,
        }
        if flip:
            candidate_amount = (
                400_000 if strategy in ("amount_only", "combined") else 500_000
            )
            candidate_term = 10 if strategy == "amount_only" else 6
            amount_change_abs = abs(candidate_amount - 500_000)
            record.update(
                {
                    "candidate_amount": candidate_amount,
                    "candidate_term": candidate_term,
                    "normalized_distance": distance,
                    "amount_change_abs": amount_change_abs,
                    "amount_change_pct": amount_change_abs / 500_000 * 100,
                    "term_change_abs": abs(candidate_term - 10),
                    "candidate_score": 0.9,
                    "score_change": 0.8,
                }
            )
        else:
            record.update(
                {
                    "candidate_amount": None,
                    "candidate_term": None,
                    "normalized_distance": None,
                    "amount_change_abs": None,
                    "amount_change_pct": None,
                    "term_change_abs": None,
                    "candidate_score": None,
                    "score_change": None,
                }
            )
        records.append(record)
    return pd.DataFrame(records, columns=_RESULT_COLUMNS)


RESULTS_FIXTURE = make_results_frame(
    [
        (1, "amount_only", True, 2, 0.10),
        (2, "amount_only", False, 10, None),
        (3, "amount_only", False, 10, None),
        (4, "amount_only", False, 10, None),
        (1, "term_only", True, 3, 0.20),
        (2, "term_only", True, 5, 0.40),
        (3, "term_only", False, 9, None),
        (4, "term_only", False, 9, None),
        (1, "combined", True, 12, 0.25),
        (2, "combined", False, 90, None),
        (3, "combined", True, 40, 0.50),
        (4, "combined", False, 90, None),
    ]
)
NO_OVERLAP_FIXTURE = make_results_frame(
    [
        (1, "amount_only", True, 2, 0.10),
        (1, "term_only", False, 9, None),
        (2, "amount_only", False, 10, None),
        (2, "term_only", True, 3, 0.20),
    ]
)
TERM_ONLY_FLIPS_FIXTURE = make_results_frame(
    [
        (1, "amount_only", False, 10, None),
        (1, "term_only", True, 3, 0.20),
        (1, "combined", False, 90, None),
    ]
)
NO_FLIPS_FIXTURE = make_results_frame(
    [
        (1, "amount_only", False, 10, None),
        (1, "term_only", False, 9, None),
        (1, "combined", False, 90, None),
    ]
)


def no_flip(strategy, n):
    return SearchResult(
        strategy, 500_000, 10, 0.1, False, n, None, None, None, None, None, None, None, None
    )


FLIP_RESULT = SearchResult(
    Strategy.TERM_ONLY, 500_000, 10, 0.1, True, 5, 500_000, 4, 6 / 18, 0, 0.0, 6, 0.9, 0.8
)
ALL_NO_FLIP = {
    Strategy.AMOUNT_ONLY: no_flip(Strategy.AMOUNT_ONLY, 10),
    Strategy.TERM_ONLY: no_flip(Strategy.TERM_ONLY, 9),
    Strategy.COMBINED: no_flip(Strategy.COMBINED, 90),
}
METRICS = {
    "accuracy": 0.97,
    "precision": 0.98,
    "recall": 0.97,
    "f1": 0.975,
    "roc_auc": 0.99,
    "confusion_matrix": [[300, 10], [15, 500]],
}


def check_args(cibil, amount=5_000_000, term=12):
    return [
        "--education",
        "Graduate",
        "--self-employed",
        "No",
        "--no-of-dependents",
        "2",
        "--income-annum",
        "4100000",
        "--cibil-score",
        str(cibil),
        "--residential-assets-value",
        "2700000",
        "--commercial-assets-value",
        "2200000",
        "--luxury-assets-value",
        "8800000",
        "--bank-asset-value",
        "3300000",
        "--loan-amount",
        str(amount),
        "--loan-term",
        str(term),
    ]


ARGS_LOW_CIBIL_LONG_TERM = check_args(400)
ARGS_APPROVED = check_args(750)
