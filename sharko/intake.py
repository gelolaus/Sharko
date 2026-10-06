from collections.abc import Callable, Mapping

from sharko.config import FEATURES
from sharko.data import SearchSpace
from sharko.messages import CURRENCY_NOTE
from sharko.plain import WIDTH, wrap


class IntakeError(ValueError):
    """An applicant field is missing or not an allowed value."""


INTAKE_FIELDS: list[str] = [
    "education",
    "self_employed",
    "no_of_dependents",
    "income_annum",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_amount",
    "loan_term",
]

_ASSET_FIELDS = {
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
}

FIELD_TITLES = {
    "education": "Education",
    "self_employed": "Self-employed",
    "no_of_dependents": "Number of dependents",
    "income_annum": "Annual income (INR)",
    "cibil_score": "CIBIL credit score",
    "residential_assets_value": "Residential assets value (INR)",
    "commercial_assets_value": "Commercial assets value (INR)",
    "luxury_assets_value": "Luxury assets value (INR)",
    "bank_asset_value": "Bank asset value (INR)",
    "loan_amount": "Loan amount (INR)",
    "loan_term": "Loan term (years)",
}


# Sample applicant for `check --example` (a Rejected application with a known fix).
EXAMPLE_APPLICANT: dict[str, str] = {
    "education": "Graduate",
    "self_employed": "No",
    "no_of_dependents": "2",
    "income_annum": "4100000",
    "cibil_score": "417",
    "residential_assets_value": "2700000",
    "commercial_assets_value": "2200000",
    "luxury_assets_value": "8800000",
    "bank_asset_value": "3300000",
    "loan_amount": "12200000",
    "loan_term": "8",
}


def _parse_whole(text: str) -> int | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def _short_title(name: str) -> str:
    """Field title without its unit, e.g. 'Loan term (years)' -> 'Loan term'."""
    return FIELD_TITLES.get(name, name).split(" (")[0]


def _reject(name: str, allowed: str) -> IntakeError:
    return IntakeError(f"{_short_title(name)} ({name}) must be {allowed}")


def field_guide(name: str, space: SearchSpace) -> str:
    """Return a short allowed-values guide for one intake field."""
    if name == "education":
        return "options: 1=Graduate, 2=Not Graduate (or type the name)"
    if name == "self_employed":
        return "options: 1=Yes, 2=No (also y/n)"
    if name == "no_of_dependents":
        return "integer >= 0"
    if name == "income_annum" or name in _ASSET_FIELDS:
        return "whole number >= 0 in INR; commas allowed (e.g. 4,100,000)"
    if name == "cibil_score":
        return "integer from 300 to 900"
    if name == "loan_amount":
        return (
            f"whole number from {space.amount_min:,} to {space.amount_max:,} "
            f"INR in steps of {space.amount_step:,}"
        )
    if name == "loan_term":
        terms = ", ".join(str(term) for term in space.terms)
        return f"one of: {terms}"
    return "see validation rules"


def field_prompt(name: str, space: SearchSpace, index: int, total: int) -> str:
    """Build the two-line interactive prompt for one field."""
    title = FIELD_TITLES.get(name, name)
    guide = field_guide(name, space)
    return f"Question {index} of {total} - {title}\n  {guide}\n> "


def validate_field(name: str, text: str, space: SearchSpace) -> int | str:
    folded = text.strip().casefold()
    if name == "education":
        if folded in {"graduate", "1"}:
            return "Graduate"
        if folded in {"not graduate", "2"}:
            return "Not Graduate"
        raise _reject(name, "Graduate or Not Graduate (1 or 2)")
    if name == "self_employed":
        if folded in {"yes", "y", "1"}:
            return "Yes"
        if folded in {"no", "n", "2"}:
            return "No"
        raise _reject(name, "Yes or No (1 or 2)")
    if name == "no_of_dependents":
        value = _parse_whole(text)
        if value is None:
            raise _reject(name, "an integer >= 0")
        return value
    if name == "income_annum" or name in _ASSET_FIELDS:
        value = _parse_whole(text)
        if value is None:
            raise _reject(name, "a whole number >= 0")
        return value
    if name == "cibil_score":
        value = _parse_whole(text)
        if value is None or value < 300 or value > 900:
            raise _reject(name, "an integer from 300 to 900")
        return value
    if name == "loan_amount":
        value = _parse_whole(text)
        allowed = (
            f"a whole number from {space.amount_min:,} to {space.amount_max:,} "
            f"in steps of {space.amount_step:,}"
        )
        if (
            value is None
            or value < space.amount_min
            or value > space.amount_max
            or value % space.amount_step != 0
        ):
            raise _reject(name, allowed)
        return value
    if name == "loan_term":
        value = _parse_whole(text)
        allowed = ", ".join(map(str, space.terms))
        if value is None or value not in space.terms:
            raise _reject(name, f"one of {allowed}")
        return value
    raise IntakeError(f"unknown field {name}")


def _print_intro(output_fn: Callable[[str], None], space: SearchSpace) -> None:
    terms = ", ".join(str(term) for term in space.terms)
    lines = [
        "",
        "=" * WIDTH,
        "SHARKO - loan what-if check",
        "=" * WIDTH,
        CURRENCY_NOTE,
        "",
        *wrap(
            f"You will answer {len(INTAKE_FIELDS)} short questions about one "
            "loan application. Sharko asks a model if it would be approved. "
            "If not, Sharko looks for the smallest change to the loan amount "
            "or term that would be approved."
        ),
        "",
        *wrap(
            "Each question shows the allowed answers. A wrong answer is "
            "explained and asked again. Press Ctrl+C at any time to quit."
        ),
        "",
        "Loan limits (learned from the training data):",
        (
            f"  amounts: {space.amount_min:,} to {space.amount_max:,} INR, "
            f"in steps of {space.amount_step:,}"
        ),
        f"  terms:   {terms} years",
        "-" * WIDTH,
    ]
    for line in lines:
        output_fn(line)


def collect_applicant(
    provided: Mapping[str, str | None],
    space: SearchSpace,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    interactive: bool = True,
) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {}
    needs_prompt = any(
        name not in provided or provided[name] is None for name in INTAKE_FIELDS
    )
    if interactive and needs_prompt:
        _print_intro(output_fn, space)

    total = len(INTAKE_FIELDS)
    for index, name in enumerate(INTAKE_FIELDS, start=1):
        if name in provided and provided[name] is not None:
            parsed[name] = validate_field(name, provided[name], space)
            continue
        if not interactive:
            flag = "--" + name.replace("_", "-")
            raise IntakeError(
                f"{_short_title(name)} is required: pass {flag}, "
                "or run without --no-prompt to be asked."
            )
        while True:
            entered = input_fn(field_prompt(name, space, index, total))
            try:
                parsed[name] = validate_field(name, entered, space)
                break
            except IntakeError as exc:
                output_fn(f"  -> {exc}")
    if interactive and needs_prompt:
        output_fn("-" * WIDTH)
        output_fn("All answers received.")
    return {feature: parsed[feature] for feature in FEATURES}
