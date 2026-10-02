import pytest

from sharko.config import FEATURES
from sharko.data import SearchSpace
from sharko.intake import IntakeError, collect_applicant, validate_field

SPACE = SearchSpace(300_000, 39_500_000, 100_000, tuple(range(2, 21, 2)))
FULL_INTAKE = {
    "education": "Graduate",
    "self_employed": "No",
    "no_of_dependents": "2",
    "income_annum": "4,100,000",
    "cibil_score": "417",
    "residential_assets_value": "2,700,000",
    "commercial_assets_value": "2,200,000",
    "luxury_assets_value": "8,800,000",
    "bank_asset_value": "3,300,000",
    "loan_amount": "12,200,000",
    "loan_term": "8",
}


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("education", "graduate", "Graduate"),
        ("education", " Not Graduate ", "Not Graduate"),
        ("self_employed", "Y", "Yes"),
        ("self_employed", "no", "No"),
        ("income_annum", "9,600,000", 9_600_000),
        ("loan_amount", "5,500,000", 5_500_000),
        ("loan_term", "12", 12),
        ("cibil_score", "300", 300),
        ("no_of_dependents", "0", 0),
    ],
)
def test_validate_field_accepts_and_normalizes(name, text, expected):
    assert validate_field(name, text, SPACE) == expected


@pytest.mark.parametrize(
    "name, text, fragment",
    [
        ("loan_amount", "5,550,000", "100,000"),
        ("loan_amount", "50,000,000", "39,500,000"),
        ("loan_amount", "100,000", "300,000"),
        ("loan_term", "7", "2, 4, 6"),
        ("cibil_score", "299", "300"),
        ("cibil_score", "901", "900"),
        ("education", "college", "Graduate"),
        ("self_employed", "maybe", "Yes"),
        ("income_annum", "-5", "0"),
        ("income_annum", "12.5", "whole"),
        ("loan_term", "abc", "loan_term"),
    ],
)
def test_validate_field_rejects_with_helpful_message(name, text, fragment):
    with pytest.raises(IntakeError) as e:
        validate_field(name, text, SPACE)
    assert fragment in str(e.value)


def test_collect_applicant_prompts_missing_and_reprompts_invalid():
    answers = iter(
        ["college", "graduate", "no"]
        + ["1"]
        + ["5,000,000"]
        + ["700"]
        + ["0"] * 4
        + ["5,500,000", "12"]
    )
    out = []
    a = collect_applicant(
        {},
        SPACE,
        input_fn=lambda _: next(answers),
        output_fn=out.append,
    )
    assert a["education"] == "Graduate" and a["loan_term"] == 12
    assert set(a) == set(FEATURES)
    assert any("Graduate" in m for m in out)


def test_collect_applicant_uses_provided_values_without_prompting():
    a = collect_applicant(FULL_INTAKE, SPACE, input_fn=lambda _: pytest.fail("prompted"))
    assert a["loan_amount"] == 12_200_000 and a["loan_term"] == 8
    assert a["education"] == "Graduate"


def test_collect_applicant_invalid_provided_value_raises_and_noninteractive_missing_raises():
    with pytest.raises(IntakeError):
        collect_applicant({**FULL_INTAKE, "loan_term": "7"}, SPACE)
    with pytest.raises(IntakeError):
        collect_applicant({}, SPACE, interactive=False)
