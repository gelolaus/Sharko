import pytest

from sharko.config import FEATURES
from sharko.data import SearchSpace
from sharko.intake import IntakeError, collect_applicant, field_prompt, validate_field

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
    prompts = []
    a = collect_applicant(
        {},
        SPACE,
        input_fn=lambda prompt: (prompts.append(prompt), next(answers))[1],
        output_fn=out.append,
    )
    assert a["education"] == "Graduate" and a["loan_term"] == 12
    assert set(a) == set(FEATURES)
    assert any("Graduate" in m for m in out)
    guide_text = "\n".join(out + prompts)
    assert "Graduate" in guide_text and "Not Graduate" in guide_text
    assert "300" in guide_text and "900" in guide_text
    assert "Indian Rupees" in guide_text or "INR" in guide_text
    assert "300,000" in guide_text and "39,500,000" in guide_text


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("education", "1", "Graduate"),
        ("education", "2", "Not Graduate"),
        ("self_employed", "1", "Yes"),
        ("self_employed", "2", "No"),
    ],
)
def test_validate_field_accepts_numbered_choices(name, text, expected):
    assert validate_field(name, text, SPACE) == expected


def test_collect_applicant_uses_provided_values_without_prompting():
    a = collect_applicant(FULL_INTAKE, SPACE, input_fn=lambda _: pytest.fail("prompted"))
    assert a["loan_amount"] == 12_200_000 and a["loan_term"] == 8
    assert a["education"] == "Graduate"


def test_collect_applicant_invalid_provided_value_raises_and_noninteractive_missing_raises():
    with pytest.raises(IntakeError):
        collect_applicant({**FULL_INTAKE, "loan_term": "7"}, SPACE)
    with pytest.raises(IntakeError):
        collect_applicant({}, SPACE, interactive=False)


def test_errors_use_friendly_name_and_keep_the_field_name():
    with pytest.raises(IntakeError) as e:
        validate_field("loan_term", "7", SPACE)
    assert "Loan term (loan_term) must be one of" in str(e.value)


def test_missing_flag_in_no_prompt_mode_names_the_flag():
    with pytest.raises(IntakeError) as e:
        collect_applicant({}, SPACE, interactive=False)
    assert "--education" in str(e.value) and "Education" in str(e.value)


def test_prompt_is_two_short_lines_with_progress_and_guide():
    prompt = field_prompt("loan_term", SPACE, 11, 11)
    first, second, last = prompt.split("\n")
    assert first == "Question 11 of 11 - Loan term (years)"
    assert "2, 4, 6" in second and last == "> "
    assert max(len(line) for line in prompt.splitlines()) <= 72


def test_intro_is_short_ascii_and_explains_the_check():
    out = []
    answers = iter(["x"])
    with pytest.raises(StopIteration):
        collect_applicant(
            {}, SPACE, input_fn=lambda _: next(answers), output_fn=out.append
        )
    text = "\n".join(out)
    assert text.isascii() and max(len(line) for line in text.splitlines()) <= 72
    assert "11 short questions" in text and "Ctrl+C" in text
    assert "300,000" in text and "39,500,000" in text
    assert len(text.splitlines()) <= 22


from sharko.intake import (  # noqa: E402
    collect_name,
    report_filename,
    unique_report_path,
)


@pytest.mark.parametrize(
    "first, last, expected",
    [
        ("Angelo", "Laus", "Angelo_Laus_Results.html"),
        ("Angelo John", "Laus", "Angelo_John_Laus_Results.html"),
        ("Jose", "Pena", "Jose_Pena_Results.html"),
        ("José", "Peña", "Jose_Pena_Results.html"),
        ("Mary-Ann", "O'Brien", "Mary-Ann_OBrien_Results.html"),
        ("  Ana  ", "de   la Cruz", "Ana_de_la_Cruz_Results.html"),
        ("李", "王", "Unnamed_Unnamed_Results.html"),
    ],
)
def test_report_filename_is_safe_and_ascii(first, last, expected):
    assert report_filename(first, last) == expected


def test_unique_report_path_never_overwrites(tmp_path):
    first = unique_report_path(tmp_path, "Ana_Cruz_Results.html")
    assert first.name == "Ana_Cruz_Results.html"
    first.write_text("x", encoding="utf-8")
    second = unique_report_path(tmp_path, "Ana_Cruz_Results.html")
    assert second.name == "Ana_Cruz_Results_2.html"
    second.write_text("x", encoding="utf-8")
    assert unique_report_path(tmp_path, "Ana_Cruz_Results.html").name == "Ana_Cruz_Results_3.html"


def test_collect_name_asks_and_reprompts_invalid_answers():
    answers = iter(["", "R2D2", "Ana Maria", "123", "dela Cruz"])
    out = []
    names = collect_name(None, None, input_fn=lambda _: next(answers), output_fn=out.append)
    assert names == ("Ana Maria", "dela Cruz")
    assert any("First name" in line and "letters" in line for line in out)


def test_collect_name_uses_given_values_and_asks_only_for_the_missing_one():
    answers = iter(["Cruz"])
    asked = []
    names = collect_name(
        "Ana", None, input_fn=lambda prompt: (asked.append(prompt), next(answers))[1],
        output_fn=lambda _: None,
    )
    assert names == ("Ana", "Cruz") and len(asked) == 1 and "Last name" in asked[0]


def test_collect_name_non_interactive_defaults_and_validates():
    assert collect_name(None, None, interactive=False) == ("Unnamed", "Applicant")
    assert collect_name("Ana", None, interactive=False) == ("Ana", "Applicant")
    with pytest.raises(IntakeError):
        collect_name("R2D2", "x", interactive=False)


def test_collect_name_rejects_very_long_names():
    with pytest.raises(IntakeError):
        collect_name("A" * 41, "Cruz", interactive=False)


def test_intro_can_be_skipped_by_the_caller():
    out = []
    answers = iter(["x"])
    with pytest.raises(StopIteration):
        collect_applicant(
            {}, SPACE, input_fn=lambda _: next(answers), output_fn=out.append,
            show_intro=False,
        )
    assert "SHARKO - loan what-if check" not in "\n".join(out)
