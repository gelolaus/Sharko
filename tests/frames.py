import numpy as np

from sharko.data import SearchSpace

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
