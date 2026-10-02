from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loan_term = rng.choice(np.arange(2, 21, 2), size=n)
    cibil_score = rng.integers(300, 901, size=n)
    residential_assets = rng.integers(0, 101, size=n) * 100_000
    residential_assets[: min(5, n)] = -100_000

    return pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1),
            "no_of_dependents": rng.integers(0, 6, size=n),
            "education": rng.choice(["Graduate", "Not Graduate"], size=n),
            "self_employed": rng.choice(["Yes", "No"], size=n),
            "income_annum": rng.integers(3, 101, size=n) * 100_000,
            "loan_amount": rng.integers(3, 101, size=n) * 100_000,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": rng.integers(0, 101, size=n) * 100_000,
            "luxury_assets_value": rng.integers(0, 101, size=n) * 100_000,
            "bank_asset_value": rng.integers(0, 101, size=n) * 100_000,
            "loan_status": np.where(
                (cibil_score >= 550) | (loan_term <= 4),
                "Approved",
                "Rejected",
            ),
        }
    )


def write_synthetic_csv(path: Path, n: int = 600, seed: int = 0) -> Path:
    frame = make_synthetic_df(n=n, seed=seed).copy()
    frame.columns = [f" {column}" for column in frame.columns]
    for column in (" education", " self_employed", " loan_status"):
        frame[column] = " " + frame[column].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path
