from collections.abc import Callable, Mapping

from sharko.config import FEATURES
from sharko.data import SearchSpace


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


def _parse_whole(text: str) -> int | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def _reject(name: str, allowed: str) -> IntakeError:
    return IntakeError(f"{name} must be {allowed}")


def validate_field(name: str, text: str, space: SearchSpace) -> int | str:
    folded = text.strip().casefold()
    if name == "education":
        if folded == "graduate":
            return "Graduate"
        if folded == "not graduate":
            return "Not Graduate"
        raise _reject(name, "Graduate or Not Graduate")
    if name == "self_employed":
        if folded in {"yes", "y"}:
            return "Yes"
        if folded in {"no", "n"}:
            return "No"
        raise _reject(name, "Yes or No")
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


def collect_applicant(
    provided: Mapping[str, str | None],
    space: SearchSpace,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    interactive: bool = True,
) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {}
    for name in INTAKE_FIELDS:
        if name in provided and provided[name] is not None:
            parsed[name] = validate_field(name, provided[name], space)
            continue
        if not interactive:
            raise IntakeError(f"{name} is required")
        while True:
            entered = input_fn(f"{name}: ")
            try:
                parsed[name] = validate_field(name, entered, space)
                break
            except IntakeError as exc:
                output_fn(str(exc))
    return {feature: parsed[feature] for feature in FEATURES}
