from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from sharko.applicant_report import ExperimentContext, write_applicant_report
from sharko.config import (
    APPROVED,
    DEFAULT_DATA,
    DEFAULT_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    DEFAULT_REPORTS,
    FEATURES,
    REJECTED,
    SEED,
)
from sharko.data import load_dataset, split_dataset
from sharko.experiment import run_experiment, summarize
from sharko.guide import start_screen, tutorial
from sharko.history import HistoryLog
from sharko.intake import (
    EXAMPLE_APPLICANT,
    INTAKE_FIELDS,
    IntakeError,
    collect_applicant,
    collect_name,
    print_intro,
    report_filename,
    unique_report_path,
)
from sharko.messages import (
    CURRENCY_NOTE,
    DISCLAIMER,
    LIMITATIONS_NOTE,
    missing_data_message,
    missing_model_message,
    missing_results_message,
)
from sharko.model import (
    approval_calibration,
    approval_scores,
    calibration_error,
    evaluate_model,
    load_bundle,
    predicted_status,
    save_bundle,
    train_model,
)
from sharko.plain import WIDTH, chance, heading, wrap
from sharko.robustness import (
    FLIP_COLUMNS,
    seed_check,
    sensitivity,
    spread,
    without_negative_assets,
)
from sharko.render import PLAIN_WAYS, STRATEGY_LABELS, format_check_result
from sharko.report import (
    CALIBRATION_FILE,
    figure_bytes,
    generate_report,
    render_calibration,
)
from sharko.scenarios import (
    PROFILE_DIMENSIONS,
    format_value,
    pick_examples,
    profile_breakdown,
)
from sharko.search import Strategy, run_all_strategies
from sharko.trace import build_trace, margin_scan

RESULTS_NAME = "experiment_results.csv"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _banner(title: str) -> None:
    print("")
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def _print_metrics(metrics: dict) -> None:
    rows = [
        ("accuracy", "accuracy", "share of loans predicted correctly"),
        ("precision", "precision", "when it says Approved, how often it is right"),
        ("recall", "recall", "share of truly Approved loans it caught"),
        ("F1", "f1", "balance of precision and recall"),
        ("ROC-AUC", "roc_auc", "ranks Approved above Rejected (1 = perfect)"),
    ]
    if "brier_score" in metrics:
        rows.append(("Brier", "brier_score", "Brier score, lower is better (0 = perfect)"))
    for label, key, meaning in rows:
        print(f"  {label:<10}{metrics[key]:.4f}  {meaning}")
    (rej_right, rej_wrong), (app_wrong, app_right) = metrics["confusion_matrix"]
    print("  Hits and misses on the test loans:")
    print(f"    Truly Rejected: {rej_right} right, {rej_wrong} wrong")
    print(f"    Truly Approved: {app_right} right, {app_wrong} wrong")


def _print_search_limits(space) -> None:
    terms = ", ".join(str(term) for term in space.terms)
    print("Loan limits learned from the data:")
    print(
        f"  amounts {space.amount_min:,} to {space.amount_max:,} INR "
        f"(steps of {space.amount_step:,}); terms {terms} years"
    )


def _print_next(command: str) -> None:
    print("")
    print(f"Next step: python -m sharko {command}")


def _print_training(bundle) -> None:
    _banner("Training complete")
    print(CURRENCY_NOTE)
    for line in wrap(
        "Sharko tried 3 kinds of model and kept the one that scored best on "
        "a practice test (cross-validation) using only the training loans."
    ):
        print(line)
    print("")
    print(f"Chosen model: {bundle.model_name}")
    print("Practice-test scores (CV AUC, 1 = perfect):")
    for name, score in bundle.cv_auc.items():
        print(f"  {name}: {score:.4f}")
    print("")
    print("Final check on loans the model never saw (20% held back):")
    _print_metrics(bundle.test_metrics)
    print("")
    _print_search_limits(bundle.search_space)
    _print_next("check --example")


def _train(args: argparse.Namespace) -> int:
    data = Path(args.data)
    model = Path(args.model)
    if not data.is_file():
        return _fail(missing_data_message(data))
    frame = load_dataset(data)
    train, test = split_dataset(frame)
    bundle = train_model(train)
    bundle.test_metrics = evaluate_model(bundle.pipeline, test)
    save_bundle(bundle, model)
    _print_training(bundle)
    return 0


def _load_split(data: Path, model: Path):
    if not data.is_file():
        return _fail(missing_data_message(data))
    if not model.is_file():
        return _fail(missing_model_message(model))
    bundle = load_bundle(model)
    frame = load_dataset(data)
    _, test = split_dataset(frame, seed=bundle.seed)
    return bundle, test


def _print_calibration(table: pd.DataFrame) -> None:
    for line in heading("Do the percentages mean what they say?"):
        print(line)
    for line in wrap(
        "Loans are grouped by the chance of approval the model gave them. "
        "If the model is reliable, the share that were truly Approved is "
        "close to what it said.",
        indent=2,
    ):
        print(line)
    print(f"  {'Chance given':<16}{'Loans':>6}{'Model said':>13}{'Truly approved':>17}")
    for row in table.itertuples():
        band = f"{row.low:.0%} to {row.high:.0%}"
        print(
            f"  {band:<16}{row.loans:>6}{row.mean_score:>13.1%}"
            f"{row.approved_rate:>17.1%}"
        )
    gap = calibration_error(table) * 100
    print(f"  Average gap between the two: {gap:.1f} percentage points.")
    for line in wrap(
        "Groups with few loans are noisy, so do not read much into one row. "
        "A model can rank loans well and still be a little over- or "
        "under-confident.",
        indent=2,
    ):
        print(line)


def _evaluate(args: argparse.Namespace) -> int:
    loaded = _load_split(Path(args.data), Path(args.model))
    if isinstance(loaded, int):
        return loaded
    bundle, test = loaded
    _banner("Evaluation")
    print(CURRENCY_NOTE)
    print("Testing the saved model on the 20% of loans it never saw...")
    _print_metrics(evaluate_model(bundle.pipeline, test))
    table = approval_calibration(bundle.pipeline, test)
    _print_calibration(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "calibration.csv", index=False, lineterminator="\n", encoding="utf-8")
    render_calibration(table, out / CALIBRATION_FILE)
    print(f"  Saved: calibration.csv, {CALIBRATION_FILE}")
    _print_next("check --example")
    print("")
    print("IMPORTANT")
    print("-" * WIDTH)
    print(DISCLAIMER)
    print(LIMITATIONS_NOTE)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    out = Path(args.out)
    loaded = _load_split(Path(args.data), Path(args.model))
    if isinstance(loaded, int):
        return loaded
    bundle, test = loaded

    def progress(done: int, total: int) -> None:
        print(f"progress {done}/{total}", file=sys.stderr)

    def scorer(frame: pd.DataFrame):
        return approval_scores(bundle.pipeline, frame)

    _banner("Batch experiment")
    print(CURRENCY_NOTE)
    for line in wrap(
        "This tests many loans at once. It takes the held-out loans that "
        "the data marks Rejected and the model also rejects (the "
        '"eligible" loans) and runs all 3 searches on each. It may take a '
        "while."
    ):
        print(line)
    results = run_experiment(
        test,
        scorer,
        bundle.search_space,
        progress=progress,
    )
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        out / RESULTS_NAME,
        index=False,
        lineterminator="\n",
        encoding="utf-8",
    )
    eligible = 0 if results.empty else int(results["loan_id"].nunique())
    print(f"Eligible loans: {eligible}")
    _print_experiment_table(summarize(results))
    print(f"Full results saved to {out / RESULTS_NAME}")
    _print_next("report --open")
    print("")
    print("IMPORTANT")
    print("-" * WIDTH)
    print(DISCLAIMER)
    print(LIMITATIONS_NOTE)
    return 0


def _print_experiment_table(summary: pd.DataFrame) -> None:
    for line in heading("Results by way of changing the loan"):
        print(line)
    if summary.empty:
        print("  No eligible loans, so there is nothing to compare.")
        return
    print(f"  {'Way':<17}{'Loans':>6}{'Approved option found':>24}{'Median tries':>14}")
    for strategy, row in summary.iterrows():
        try:
            way = PLAIN_WAYS[Strategy(strategy)]
        except ValueError:
            way = str(strategy)
        found = row["flip_rate_pct"]
        tries = row["median_configs_evaluated"]
        found_text = "n/a" if pd.isna(found) else f"{found:.1f}%"
        tries_text = "n/a" if pd.isna(tries) else f"{tries:,.0f}"
        print(
            f"  {way:<17}{int(row['eligible']):>6}"
            f"{found_text:>24}{tries_text:>14}"
        )
    print("")
    print("How to read this table")
    for text in (
        "Loans: how many loans the search was run on.",
        "Approved option found: share of those loans where the search "
        "found a change the model would approve.",
        "Median tries: the typical number of options checked before "
        "stopping. Fewer means a quicker answer.",
    ):
        for line in wrap(f"- {text}", indent=2, hang=2):
            print(line)
    print("")


def _score_applicant(scorer, applicant) -> float:
    frame = pd.DataFrame(
        [{feature: applicant[feature] for feature in FEATURES}],
        columns=FEATURES,
    )
    return float(np.asarray(scorer(frame), dtype=float).reshape(-1)[0])


def _check(args: argparse.Namespace) -> int:
    model = Path(args.model)
    if not model.is_file():
        return _fail(missing_model_message(model))
    bundle = load_bundle(model)
    provided = {name: getattr(args, name) for name in INTAKE_FIELDS}
    if args.example:
        provided = {
            name: value if value is not None else EXAMPLE_APPLICANT[name]
            for name, value in provided.items()
        }
        print("")
        for line in wrap(
            "Using the built-in example application. Change any answer with "
            "its flag, for example --cibil-score 800."
        ):
            print(line)
    interactive = not args.no_prompt
    space = bundle.search_space
    try:
        if interactive and any(value is None for value in provided.values()):
            print_intro(print, space)
        first, last = collect_name(
            args.first_name or ("Example" if args.example else None),
            args.last_name or ("Applicant" if args.example else None),
            input_fn=input,
            interactive=interactive and not args.example,
        )
        applicant = collect_applicant(
            provided,
            space,
            input_fn=input,
            interactive=interactive,
            show_intro=False,
        )
    except IntakeError as exc:
        hint = ""
        if args.example:
            hint = (
                "\nThe built-in example was made for the default dataset. "
                "Override a value with its flag."
            )
        return _fail(str(exc) + hint)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. Nothing was saved.", file=sys.stderr)
        return 130

    scorer = lambda df: approval_scores(bundle.pipeline, df)
    print("Checking the application...")
    baseline_score = _score_applicant(scorer, applicant)
    if predicted_status(baseline_score) == APPROVED:
        text = format_check_result(
            applicant,
            baseline_score,
            None,
            bundle.test_metrics,
            bundle.search_space,
        )
        strategies = None
        results = None
        trace = None
        margin = margin_scan(applicant, scorer, bundle.search_space)
        status = APPROVED
    else:
        print(
            "Not approved as submitted. Looking for the smallest "
            "approved change..."
        )
        baseline_score, results = run_all_strategies(
            applicant,
            scorer,
            bundle.search_space,
        )
        text = format_check_result(
            applicant,
            baseline_score,
            results,
            bundle.test_metrics,
            bundle.search_space,
        )
        strategies = [results[strategy].as_dict() for strategy in Strategy]
        status = REJECTED
        trace = build_trace(applicant, scorer, bundle.search_space, results)
        margin = None
    print(text)
    _save_report(
        args,
        (first, last),
        applicant,
        baseline_score,
        results,
        trace,
        margin,
        _experiment_context(Path(args.out)),
        bundle,
    )
    HistoryLog(Path(args.history)).append(
        {
            "source": "example" if args.example else "interactive",
            "applicant": {name: applicant[name] for name in FEATURES},
            "baseline_score": float(baseline_score),
            "baseline_status": status,
            "strategies": strategies,
        }
    )
    return 0


def _experiment_context(out: Path) -> ExperimentContext | None:
    """The experiment's charts and summary for the report, or None if it was not run."""
    results_path = out / RESULTS_NAME
    if not results_path.is_file():
        return None
    try:
        frame = pd.read_csv(results_path, encoding="utf-8")
        eligible = int(frame["loan_id"].nunique()) if len(frame) else 0
        return ExperimentContext(
            figure_bytes(frame, out, results_path), summarize(frame), eligible
        )
    except (OSError, ValueError, KeyError):
        return None


def _save_report(
    args, names, applicant, baseline_score, results, trace, margin, experiment, bundle
) -> None:
    """Write the per-person HTML report, say where it is, and optionally open it."""
    first, last = names
    folder = Path(args.reports_dir)
    try:
        path = write_applicant_report(
            unique_report_path(folder, report_filename(first, last)),
            first,
            last,
            applicant,
            baseline_score,
            results,
            trace,
            bundle.test_metrics,
            bundle.search_space,
            experiment=experiment,
            margin=margin,
        )
    except OSError as exc:
        print(f"Could not save the report: {exc}", file=sys.stderr)
        return
    for line in heading("REPORT SAVED"):
        print(line)
    contents = (
        "Your report has your answers, the result, and every try the search "
        "made, step by step."
        if results is not None
        else "Your report has your answers, the result, and charts showing "
        "how safe the approval is."
    )
    if experiment is not None:
        contents += " It also has the paper's charts for all the test loans."
    for line in wrap(contents, indent=2):
        print(line)
    print(f"  Folder: {path.parent.resolve()}")
    print(f"  File:   {path.name}")
    print("  Open it by double-clicking the file, or in any web browser.")
    if experiment is None:
        for line in wrap(
            "Tip: run python -m sharko experiment, then check again, to add "
            "the paper's charts to your report.",
            indent=2,
        ):
            print(line)
    asks = not args.no_prompt and not args.example
    if args.open:
        _open_page(path)
    elif asks:
        try:
            answer = input("Open it in your browser now? (y/n) ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer.strip().lower() in {"y", "yes"}:
            _open_page(path)
    else:
        print("  Tip: add --open to open it automatically next time.")


def _open_page(path: Path) -> None:
    print("Opening the report in your browser...")
    if not webbrowser.open(path.resolve().as_uri()):
        print(f"Could not open a browser. Open this file yourself: {path}")


def _closest_logged(record: dict) -> str | None:
    """Plain name of the way that found the closest approved option, if any."""
    if record.get("baseline_status") != REJECTED:
        return None
    logged = record.get("strategies")
    if not logged:
        return None
    by_name = {item["strategy"]: item for item in logged}
    closest = None
    closest_distance = None
    for strategy in Strategy:
        item = by_name.get(strategy.value)
        if item is None or not item.get("flip_found"):
            continue
        distance = item.get("normalized_distance")
        if distance is None:
            continue
        if closest is None or distance < closest_distance:
            closest = item
            closest_distance = distance
    if closest is None:
        return None
    return PLAIN_WAYS[Strategy(closest["strategy"])]


def _history(args: argparse.Namespace) -> int:
    records = HistoryLog(Path(args.history)).read()
    if not records:
        print("No history yet. Run: python -m sharko check --example")
        return 0
    _banner("Check history")
    print(f"Runs logged: {len(records)}")
    print(CURRENCY_NOTE)
    for line in wrap(
        "Times are UTC. chance = the model's approval chance. fix = the "
        "closest change found that would be approved."
    ):
        print(line)
    print("-" * WIDTH)
    last = args.last
    window = records[-last:] if last else []
    for record in window:
        when = str(record["timestamp"])[:16].replace("T", " ")
        line = (
            f"{when}  {record['baseline_status']:<8}  "
            f"chance {chance(record['baseline_score'])}"
        )
        closest = _closest_logged(record)
        if closest is not None:
            line += f"  fix: {closest}"
        print(line)
    return 0


_REPORT_FILES = {
    "summary.csv": "one row per way, with all the averages",
    "paired_comparison.csv": "ways compared on the loans both could fix",
    "flip_rate_by_strategy.png": "how often each way found an approved option",
    "distance_by_strategy.png": "how big a change each way needed",
    "configs_evaluated_by_strategy.png": "how many options each way tried",
    "amount_original_vs_flip.png": "original loan amount vs the new one",
}


def _report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    results_path = out / RESULTS_NAME
    if not results_path.is_file():
        return _fail(missing_results_message(results_path))
    _banner("Report")
    print("Writing summary tables and figures...")
    results = pd.read_csv(results_path, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    paths = generate_report(results, out, reports_dir=reports_dir)
    print(f"Tables and charts saved in {out}:")
    page = reports_dir / "Experiment_Report.html"
    for path in paths:
        if path != page:
            print(f"  {path.name}")
            meaning = _REPORT_FILES.get(path.name)
            if meaning:
                print(f"      {meaning}")
    print("")
    print(f"All charts on one page (open it in a browser), saved in {reports_dir}:")
    print(f"  {page.name}")
    if args.open:
        _open_page(page)
    else:
        print("To see the charts: python -m sharko report --open")
    return 0


def _flip_text(row) -> str:
    parts = []
    for strategy, column in zip(Strategy, FLIP_COLUMNS):
        value = row[column]
        text = "n/a" if pd.isna(value) else f"{value:.1f}%"
        parts.append(f"{PLAIN_WAYS[strategy].lower()} {text}")
    return ", ".join(parts)


def _print_sensitivity(table: pd.DataFrame) -> None:
    for line in heading("1. With and without the loans that have a negative asset value"):
        print(line)
    for name, row in table.iterrows():
        print(f"  {name} ({int(row['records']):,} loans, {row['model']})")
        print(
            f"    accuracy {row['accuracy']:.4f}  F1 {row['f1']:.4f}  "
            f"ROC-AUC {row['roc_auc']:.4f}"
        )
        for line in wrap(
            f"approved option found on {int(row['eligible'])} loans: "
            + _flip_text(row),
            indent=4,
            hang=2,
        ):
            print(line)


def _print_seeds(table: pd.DataFrame) -> None:
    for line in heading("2. The same run with different random splits (seeds)"):
        print(line)
    print(
        f"  {'Seed':>5}{'Accuracy':>10}{'ROC-AUC':>9}"
        f"{'Amount only':>13}{'Term only':>11}{'Both':>7}"
    )
    for seed, row in table.iterrows():
        flips = [row[column] for column in FLIP_COLUMNS]
        print(
            f"  {seed:>5}{row['accuracy']:>10.4f}{row['roc_auc']:>9.4f}"
            f"{flips[0]:>12.1f}%{flips[1]:>10.1f}%{flips[2]:>6.1f}%"
        )
    stats = spread(table)
    print("")
    print("  Across all seeds (approved option found, in %):")
    for label, column in zip(("amount only", "term only", "both"), FLIP_COLUMNS):
        print(
            f"    {label:<12} mean {stats.loc['mean', column]:.1f}, "
            f"std {stats.loc['std', column]:.1f}, "
            f"range {stats.loc['min', column]:.1f} to {stats.loc['max', column]:.1f}"
        )
    print(
        f"    accuracy     mean {stats.loc['mean', 'accuracy']:.4f}, "
        f"std {stats.loc['std', 'accuracy']:.4f}"
    )


def _robustness(args: argparse.Namespace) -> int:
    data = Path(args.data)
    if not data.is_file():
        return _fail(missing_data_message(data))
    if args.seeds < 2:
        return _fail("--seeds must be 2 or more, so the runs can be compared.")
    frame = load_dataset(data)
    negatives = len(frame) - len(without_negative_assets(frame))
    seeds = [args.first_seed + offset for offset in range(args.seeds)]
    _banner("Robustness checks")
    print(CURRENCY_NOTE)
    for line in wrap(
        "Each run trains the model again and runs all 3 searches, so this "
        f"takes a few minutes in all ({args.seeds + 1} runs here). "
        f"{negatives} loans in the data have a negative residential asset "
        "value. They are normally kept as they are."
    ):
        print(line)
    print("Working...", file=sys.stderr)
    sensitive = sensitivity(frame)
    seeded = seed_check(frame, seeds)
    _print_sensitivity(sensitive)
    _print_seeds(seeded)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sensitive.to_csv(out / "robustness_sensitivity.csv", lineterminator="\n", encoding="utf-8")
    seeded.to_csv(out / "robustness_seeds.csv", lineterminator="\n", encoding="utf-8")
    print("")
    print(f"Saved to {out}: robustness_sensitivity.csv, robustness_seeds.csv")
    print("")
    for line in wrap(
        "How to read this: if the numbers barely move, the results do not "
        "depend on one lucky split or on those few loans. A big move would "
        "be a warning. A high chance of approval is still a model estimate "
        "and never a promise that a lender will approve a loan."
    ):
        print(line)
    print("")
    print("IMPORTANT")
    print("-" * WIDTH)
    print(DISCLAIMER)
    print(LIMITATIONS_NOTE)
    return 0


def _example_lines(label: str, rows: pd.DataFrame, loan: pd.Series) -> list[str]:
    lines = heading(f"{label}")
    lines.extend(
        wrap(
            f"CIBIL {int(loan['cibil_score'])}, income "
            f"{int(loan['income_annum']):,} INR, asked for "
            f"{int(loan['loan_amount']):,} INR over {int(loan['loan_term'])} years.",
            indent=2,
        )
    )
    lines.append(f"  Approval chance as asked: {chance(rows.iloc[0]['baseline_score'])}.")
    lines.append(f"  {'Way':<17}{'New amount':>14}{'Term':>8}{'Chance':>9}{'Distance':>10}")
    for strategy in Strategy:
        row = rows.loc[rows["strategy"] == strategy.value].iloc[0]
        way = PLAIN_WAYS[strategy]
        if not bool(row["flip_found"]):
            lines.append(f"  {way:<17}{'none found':>14}")
            continue
        lines.append(
            f"  {way:<17}{int(row['candidate_amount']):>14,}"
            f"{int(row['candidate_term']):>7}y{chance(row['candidate_score']):>9}"
            f"{row['normalized_distance']:>10.3f}"
        )
    return lines


def _print_breakdown(table: pd.DataFrame) -> None:
    for key, title in PROFILE_DIMENSIONS:
        part = table[table["dimension"] == title]
        if part.empty:
            continue
        for line in heading(f"{title}: approved option found, % of loans"):
            print(line)
        print(f"  {'Group (range)':<30}{'Loans':>6}{'Amount only':>13}{'Term only':>11}{'Both':>8}")
        for band, group in part.groupby("band", sort=False):
            rate = group.set_index("strategy")["flip_rate_pct"]
            first = group.iloc[0]
            text = (
                f"{band} ({format_value(key, first['low'])}-"
                f"{format_value(key, first['high'])})"
            )
            print(
                f"  {text:<30}{int(first['eligible']):>6}"
                f"{rate.get('amount_only', float('nan')):>12.1f}%"
                f"{rate.get('term_only', float('nan')):>10.1f}%"
                f"{rate.get('combined', float('nan')):>7.1f}%"
            )


def _scenarios(args: argparse.Namespace) -> int:
    loaded = _load_split(Path(args.data), Path(args.model))
    if isinstance(loaded, int):
        return loaded
    bundle, test = loaded

    def scorer(frame: pd.DataFrame):
        return approval_scores(bundle.pipeline, frame)

    _banner("Scenarios across applicant profiles")
    print(CURRENCY_NOTE)
    for line in wrap(
        "Sharko takes the held-out loans that the data marks Rejected and "
        "the model also rejects, runs all 3 searches on each, and then "
        "compares the 3 ways for different kinds of applicant. It may take "
        "a little while."
    ):
        print(line)
    results = run_experiment(test, scorer, bundle.search_space)
    if results.empty:
        print("No eligible loans, so there are no scenarios to compare.")
        return 0
    loans = test.set_index("loan_id")
    for label, loan_id in pick_examples(results, test):
        rows = results.loc[results["loan_id"] == loan_id]
        for line in _example_lines(label, rows, loans.loc[loan_id]):
            print(line)
    table = profile_breakdown(results, test)
    _print_breakdown(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scenarios.csv", index=False, lineterminator="\n", encoding="utf-8")
    print("")
    print(f"Saved to {out / 'scenarios.csv'}")
    for line in wrap(
        "How to read this: a bigger gap between the 3 ways, or between "
        "groups, shows where the model is most sensitive to the loan "
        "amount and term. It is a pattern in past data, not advice."
    ):
        print(line)
    print("")
    print("IMPORTANT")
    print("-" * WIDTH)
    print(DISCLAIMER)
    print(LIMITATIONS_NOTE)
    return 0


def _tutorial(args: argparse.Namespace) -> int:
    print(tutorial())
    return 0


_INTAKE_HELP = {
    "education": "Graduate or Not Graduate (also 1/2)",
    "self_employed": "Yes or No (also y/n or 1/2)",
    "no_of_dependents": "integer >= 0",
    "income_annum": "annual income in INR (whole number)",
    "cibil_score": "CIBIL score from 300 to 900",
    "residential_assets_value": "residential assets in INR",
    "commercial_assets_value": "commercial assets in INR",
    "luxury_assets_value": "luxury assets in INR",
    "bank_asset_value": "bank assets in INR",
    "loan_amount": "loan amount in INR (training-grid multiple)",
    "loan_term": "loan term in years from the training set",
}


def _add_data_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="path to loan_approval_dataset.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="path to the saved model bundle",
    )


def _add_intake_flags(parser: argparse.ArgumentParser) -> None:
    for name in INTAKE_FIELDS:
        flag = "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, default=None, help=_INTAKE_HELP[name])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharko",
        description=(
            "Sharko predicts whether a loan application would be approved. "
            "If it would be rejected, Sharko finds the smallest change to the "
            "loan amount or term that would be approved. " + CURRENCY_NOTE
        ),
        epilog="New here? Run: python -m sharko tutorial",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    train = sub.add_parser(
        "train",
        help="step 1: teach Sharko from the loan data (run once)",
        description=(
            "Teach Sharko from the loan data. It tries 3 kinds of model, "
            "keeps the best one and saves it. Run this once."
        ),
    )
    _add_data_model(train)
    train.set_defaults(func=_train)

    evaluate = sub.add_parser(
        "evaluate",
        help="show how accurate the saved model is",
        description="Test the saved model on loans it never saw and print the scores.",
    )
    _add_data_model(evaluate)
    evaluate.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for calibration.csv and calibration.png",
    )
    evaluate.set_defaults(func=_evaluate)

    check = sub.add_parser(
        "check",
        help="check one loan application (try: check --example)",
        description=(
            "Check one loan application. Answer the questions, or pass the "
            "answers as flags. If it is not approved, Sharko finds the "
            "smallest change to the loan amount or term that would be "
            "approved. Use --example to see a worked example. " + CURRENCY_NOTE
        ),
    )
    check.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="path to the saved model bundle",
    )
    check.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="append-only JSONL log path",
    )
    check.add_argument("--first-name", default=None, help="first name for the report title and file name")
    check.add_argument("--last-name", default=None, help="last name for the report title and file name")
    check.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="folder with the experiment results, for the paper's charts (default: outputs)",
    )
    check.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS,
        help="folder for the HTML report (default: Reports)",
    )
    check.add_argument(
        "--open",
        action="store_true",
        help="open the report in your browser when it is saved",
    )
    check.add_argument(
        "--example",
        action="store_true",
        help="use a built-in sample application (flags still override it)",
    )
    check.add_argument(
        "--no-prompt",
        action="store_true",
        help="do not ask for missing fields; require every applicant flag",
    )
    _add_intake_flags(check)
    check.set_defaults(func=_check)

    experiment = sub.add_parser(
        "experiment",
        help="for analysts: run the what-if search on many loans",
        description=(
            "Run all 3 searches (amount only, term only, both) on every "
            "eligible held-out loan. " + CURRENCY_NOTE
        ),
    )
    _add_data_model(experiment)
    experiment.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for experiment_results.csv",
    )
    experiment.set_defaults(func=_experiment)

    robustness = sub.add_parser(
        "robustness",
        help="for analysts: test how steady the results are",
        description=(
            "Compare the results with and without the loans that have a "
            "negative residential asset value, and repeat the run with "
            "different random splits. Takes a few minutes. " + CURRENCY_NOTE
        ),
    )
    robustness.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="path to loan_approval_dataset.csv",
    )
    robustness.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="how many random splits to try (default: 5)",
    )
    robustness.add_argument(
        "--first-seed",
        type=int,
        default=SEED,
        help="the first random seed to try; the others follow it (default: 42)",
    )
    robustness.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for the two robustness CSV files",
    )
    robustness.set_defaults(func=_robustness)

    scenarios = sub.add_parser(
        "scenarios",
        help="for analysts: compare the 3 ways across applicant profiles",
        description=(
            "Show worked examples and compare the 3 ways of changing the loan "
            "for low, middle and high CIBIL score, income and loan size. "
            + CURRENCY_NOTE
        ),
    )
    _add_data_model(scenarios)
    scenarios.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for scenarios.csv",
    )
    scenarios.set_defaults(func=_scenarios)

    report = sub.add_parser(
        "report",
        help="for analysts: draw charts and tables from the experiment",
        description="Build summary tables and charts from the experiment results.",
    )
    report.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory containing experiment_results.csv",
    )
    report.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS,
        help="folder for the charts page (default: Reports)",
    )
    report.add_argument(
        "--open",
        action="store_true",
        help="open the dashboard (report.html) in your browser",
    )
    report.set_defaults(func=_report)

    history = sub.add_parser(
        "history",
        help="list your earlier checks",
        description="Show your most recent checks. Nothing here changes a later result.",
    )
    history.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="JSONL history path",
    )
    history.add_argument(
        "--last",
        type=int,
        default=10,
        help="how many recent runs to print (default: 10)",
    )
    history.set_defaults(func=_history)

    guide = sub.add_parser(
        "tutorial",
        help="step-by-step guide to installing, running and reading Sharko",
        description="Print a short step-by-step guide to using Sharko.",
    )
    guide.set_defaults(func=_tutorial)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    if args.command is None:
        print(start_screen())
        return 0
    return args.func(args)
