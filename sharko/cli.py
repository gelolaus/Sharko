from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sharko.config import (
    APPROVED,
    DEFAULT_DATA,
    DEFAULT_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    FEATURES,
    REJECTED,
)
from sharko.data import load_dataset, split_dataset
from sharko.experiment import run_experiment, summarize
from sharko.history import HistoryLog
from sharko.intake import INTAKE_FIELDS, IntakeError, collect_applicant
from sharko.messages import (
    DISCLAIMER,
    LIMITATIONS_NOTE,
    missing_data_message,
    missing_model_message,
    missing_results_message,
)
from sharko.model import (
    approval_scores,
    evaluate_model,
    load_bundle,
    predicted_status,
    save_bundle,
    train_model,
)
from sharko.render import STRATEGY_LABELS, format_check_result
from sharko.report import generate_report
from sharko.search import Strategy, run_all_strategies

RESULTS_NAME = "experiment_results.csv"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _print_metrics(metrics: dict) -> None:
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"confusion matrix: {metrics['confusion_matrix']}")


def _print_training(bundle) -> None:
    print(f"chosen model: {bundle.model_name}")
    print("CV AUC:")
    for name, score in bundle.cv_auc.items():
        print(f"{name}: {score:.4f}")
    _print_metrics(bundle.test_metrics)


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


def _evaluate(args: argparse.Namespace) -> int:
    loaded = _load_split(Path(args.data), Path(args.model))
    if isinstance(loaded, int):
        return loaded
    bundle, test = loaded
    _print_metrics(evaluate_model(bundle.pipeline, test))
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
    print(f"eligible: {eligible}")
    print(summarize(results).to_string())
    print(DISCLAIMER)
    print(LIMITATIONS_NOTE)
    return 0


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
    try:
        applicant = collect_applicant(
            provided,
            bundle.search_space,
            interactive=not args.no_prompt,
        )
    except IntakeError as exc:
        return _fail(str(exc))

    scorer = lambda df: approval_scores(bundle.pipeline, df)
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
        status = APPROVED
    else:
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
    print(text)
    HistoryLog(Path(args.history)).append(
        {
            "source": "interactive",
            "applicant": {name: applicant[name] for name in FEATURES},
            "baseline_score": float(baseline_score),
            "baseline_status": status,
            "strategies": strategies,
        }
    )
    return 0


def _closest_logged(record: dict) -> tuple[str, int] | None:
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
    label = STRATEGY_LABELS[Strategy(closest["strategy"])]
    return label, int(closest["configurations_evaluated"])


def _history(args: argparse.Namespace) -> int:
    records = HistoryLog(Path(args.history)).read()
    if not records:
        print("No history yet.")
        return 0
    print(f"Runs logged: {len(records)}")
    last = args.last
    window = records[-last:] if last else []
    for record in window:
        line = (
            f"{record['timestamp']} | {record['baseline_status']} | "
            f"score {float(record['baseline_score']):.3f}"
        )
        closest = _closest_logged(record)
        if closest is not None:
            label, evaluated = closest
            line += f" | closest: {label} | configurations evaluated: {evaluated}"
        print(line)
    return 0


def _report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    results_path = out / RESULTS_NAME
    if not results_path.is_file():
        return _fail(missing_results_message(results_path))
    results = pd.read_csv(results_path, encoding="utf-8")
    for path in generate_report(results, out):
        print(path)
    return 0


def _add_data_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)


def _add_intake_flags(parser: argparse.ArgumentParser) -> None:
    for name in INTAKE_FIELDS:
        flag = "--" + name.replace("_", "-")
        parser.add_argument(flag, dest=name, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sharko")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a model and save the bundle")
    _add_data_model(train)
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("evaluate", help="recompute saved test metrics")
    _add_data_model(evaluate)
    evaluate.set_defaults(func=_evaluate)

    check = sub.add_parser("check", help="score one application and search flips")
    check.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    check.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    check.add_argument("--no-prompt", action="store_true")
    _add_intake_flags(check)
    check.set_defaults(func=_check)

    experiment = sub.add_parser("experiment", help="search flips on the test split")
    _add_data_model(experiment)
    experiment.add_argument("--out", type=Path, default=DEFAULT_OUT)
    experiment.set_defaults(func=_experiment)

    report = sub.add_parser("report", help="write tables and figures")
    report.add_argument("--out", type=Path, default=DEFAULT_OUT)
    report.set_defaults(func=_report)

    history = sub.add_parser("history", help="list logged check runs")
    history.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    history.add_argument("--last", type=int, default=10)
    history.set_defaults(func=_history)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    return args.func(args)
