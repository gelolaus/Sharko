from pathlib import Path

SEED = 42
APPROVAL_THRESHOLD = 0.5
AMOUNT_STEP = 100_000

APPROVED = "Approved"
REJECTED = "Rejected"
TARGET = "loan_status"
ID_COLUMN = "loan_id"

CATEGORICAL_FEATURES = ["education", "self_employed"]
NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FIXED_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in ("loan_amount", "loan_term")
]

DEFAULT_DATA = Path("loan_approval_dataset.csv")
DEFAULT_MODEL = Path("artifacts/model.joblib")
DEFAULT_HISTORY = Path("artifacts/history.jsonl")
DEFAULT_OUT = Path("outputs")
DEFAULT_REPORTS = Path("Reports")
