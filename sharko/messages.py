from pathlib import Path

DISCLAIMER = (
    "Model-generated what-if result. This is not a lender decision, "
    "a loan offer, or financial advice."
)
LIMITATIONS_NOTE = (
    "Limitations: the model reflects patterns in a public dataset that lacks "
    "economic variables (e.g. inflation, interest rates) and demographic detail; "
    "a missing flip means none was found within the searched amounts and terms, "
    "not that none exists."
)


def missing_model_message(path: Path) -> str:
    return f"Model not found at {path}. Run: python -m sharko train"


def missing_data_message(path: Path) -> str:
    return (
        f"Dataset not found at {path}. "
        "Pass --data <path to loan_approval_dataset.csv>."
    )


def missing_results_message(path: Path) -> str:
    return (
        f"Experiment results not found at {path}. "
        "Run: python -m sharko experiment"
    )
