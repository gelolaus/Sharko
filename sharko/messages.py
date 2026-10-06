from pathlib import Path

# Both notes are hard-wrapped to fit 72 columns so every screen prints them whole.
DISCLAIMER = (
    "Model-generated what-if result. This is not a lender decision,\n"
    "a loan offer, or financial advice."
)
LIMITATIONS_NOTE = (
    "Limitations: the model reflects patterns in a public dataset that\n"
    "lacks economic variables (e.g. inflation, interest rates) and\n"
    "demographic detail. If no approved option is found, that means none\n"
    "was found within the searched amounts and terms, not that none exists."
)
CURRENCY_NOTE = (
    "The amounts used here are in Indian Rupees (INR)."
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
