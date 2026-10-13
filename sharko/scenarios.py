import pandas as pd

from sharko.experiment import summarize
from sharko.search import Strategy

# (column or derived name, title shown to the reader)
PROFILE_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("cibil_score", "CIBIL credit score"),
    ("income_annum", "Annual income (INR)"),
    ("loan_to_income", "Loan amount / annual income"),
)
BANDS = ("lowest third", "middle third", "highest third")
BREAKDOWN_COLUMNS = [
    "dimension",
    "band",
    "low",
    "high",
    "strategy",
    "eligible",
    "flips",
    "flip_rate_pct",
    "mean_distance",
]
_SUMMARY_PICK = ["eligible", "flips", "flip_rate_pct", "mean_distance"]


def _eligible_loans(results: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    loans = test.loc[test["loan_id"].isin(results["loan_id"])].copy()
    loans["loan_to_income"] = loans["loan_amount"] / loans["income_annum"]
    return loans


def _bands(values: pd.Series) -> pd.Series:
    """Label each loan as the lowest, middle or highest third of its own profile value."""
    if len(values) < 3:
        return pd.Series("lowest third", index=values.index)
    return pd.qcut(values.rank(method="first"), 3, labels=list(BANDS)).astype(str)


def profile_breakdown(results: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Strategy results for each third of the eligible loans, per applicant profile."""
    rows = []
    loans = _eligible_loans(results, test)
    for key, title in PROFILE_DIMENSIONS:
        if loans.empty:
            break
        bands = _bands(loans[key])
        for band in BANDS:
            members = loans.loc[bands == band]
            if members.empty:
                continue
            summary = summarize(results.loc[results["loan_id"].isin(members["loan_id"])])
            for strategy in Strategy:
                if strategy.value not in summary.index:
                    continue
                picked = summary.loc[strategy.value, _SUMMARY_PICK]
                rows.append(
                    {
                        "dimension": title,
                        "band": band,
                        "low": members[key].min(),
                        "high": members[key].max(),
                        "strategy": strategy.value,
                        **picked.to_dict(),
                    }
                )
    return pd.DataFrame(rows, columns=BREAKDOWN_COLUMNS)


def pick_examples(results: pd.DataFrame, test: pd.DataFrame) -> list[tuple[str, int]]:
    """Four worked examples taken from the eligible held-out loans, chosen by profile."""
    loans = _eligible_loans(results, test)
    if loans.empty:
        return []
    by_cibil = loans.sort_values(["cibil_score", "loan_id"])
    by_ratio = loans.sort_values(["loan_to_income", "loan_id"])
    return [
        ("Lowest CIBIL score", int(by_cibil.iloc[0]["loan_id"])),
        ("Middle CIBIL score", int(by_cibil.iloc[(len(by_cibil) - 1) // 2]["loan_id"])),
        ("Highest CIBIL score", int(by_cibil.iloc[-1]["loan_id"])),
        ("Largest loan for the income", int(by_ratio.iloc[-1]["loan_id"])),
    ]


def format_value(key: str, value: float) -> str:
    """Short text for a band edge: INR incomes in millions so a row fits 72 columns."""
    if key == "loan_to_income":
        return f"{value:.2f}"
    if key == "income_annum":
        return f"{value / 1_000_000:.1f}M"
    return f"{value:,.0f}"
