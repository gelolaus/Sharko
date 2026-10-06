from sharko.messages import CURRENCY_NOTE
from sharko.plain import WIDTH, wrap

_BAR = "=" * WIDTH
_RULE = "-" * WIDTH

_WHAT = (
    "Sharko predicts whether a loan application would be approved. If it "
    "would be rejected, Sharko looks for the smallest change to the loan "
    "amount or the loan term that would be approved."
)


def start_screen() -> str:
    lines = [
        "",
        _BAR,
        "SHARKO - loan what-if tool",
        _BAR,
        *wrap(_WHAT),
        CURRENCY_NOTE,
        "",
        "QUICK START (2 steps)",
        _RULE,
        "  1. python -m sharko train",
        "       Teach Sharko from the loan data. Run this once.",
        "  2. python -m sharko check --example",
        "       See a full worked example, no typing needed.",
        "",
        "THEN",
        _RULE,
        "  python -m sharko check      Enter your own application",
        "  python -m sharko tutorial   Step-by-step guide",
        "  python -m sharko history    Look back at earlier checks",
        "",
        "All commands: train, evaluate, check, history, experiment, report,",
        "tutorial. Add --help to any command for details.",
    ]
    return "\n".join(lines)


def tutorial() -> str:
    lines = [
        "",
        _BAR,
        "SHARKO TUTORIAL - how to install, run and read it",
        _BAR,
        *wrap(_WHAT),
        CURRENCY_NOTE,
        "",
        "Step 0 - Install (once)",
        _RULE,
        "  python -m pip install -r requirements.txt",
        *wrap("Needs Python 3.10 or newer. Run it from the project folder.", 2),
        "",
        "Step 1 - Teach Sharko (once)",
        _RULE,
        "  python -m sharko train",
        *wrap(
            "Sharko learns from loan_approval_dataset.csv and saves what it "
            "learned to artifacts/model.joblib. Run it again only if the "
            "data changes.",
            2,
        ),
        "",
        "Step 2 - Try the built-in example",
        _RULE,
        "  python -m sharko check --example",
        *wrap(
            "Shows a full result for a sample application, so you can see "
            "what the output looks like before typing anything.",
            2,
        ),
        "",
        "Step 3 - Check your own application",
        _RULE,
        "  python -m sharko check",
        *wrap(
            "Answer 11 short questions. Each one shows the allowed answers; "
            "type the number or the name. A wrong answer is explained and "
            "asked again.",
            2,
        ),
        "",
        "Step 4 - Read the result, top to bottom",
        _RULE,
        "  RESULT                  Approved or not, with the model's chance",
        "  SMALLEST CHANGE         The closest loan amount/term that is approved",
        "  HOW SHARKO FOUND THIS   The 4 steps behind the answer",
        "  THE 3 WAYS COMPARED     Amount only, term only, or both",
        "  IMPORTANT               What this is not (not a lender decision)",
        "  TECHNICAL DETAILS       Exact numbers. Optional, for analysts.",
        "",
        "Step 5 - Look back",
        _RULE,
        "  python -m sharko history",
        "  Lists your earlier checks. Nothing in it changes a later result.",
        "",
        "Optional - for analysts",
        _RULE,
        "  python -m sharko evaluate    How accurate the model is",
        "  python -m sharko experiment  Run the what-if search on many loans",
        "  python -m sharko report      Draw charts from the experiment",
        "",
        "Troubleshooting",
        _RULE,
        "  Model not found      Run: python -m sharko train",
        "  Dataset not found    Run from the project folder, or pass",
        "                       --data <path to loan_approval_dataset.csv>",
        "  ... must be one of   Use a value from the list under the question",
        "  Want to stop         Press Ctrl+C",
    ]
    return "\n".join(lines)
