import pandas as pd

from sharko.config import SEED
from sharko.data import split_dataset
from sharko.experiment import run_experiment, summarize
from sharko.model import approval_scores, evaluate_model, train_model
from sharko.search import Strategy

ASSET_COLUMN = "residential_assets_value"
METRIC_COLUMNS = ["accuracy", "f1", "roc_auc"]
FLIP_COLUMNS = [f"{strategy.value}_flip_pct" for strategy in Strategy]
RUN_COLUMNS = ["records", "model", *METRIC_COLUMNS, "eligible", *FLIP_COLUMNS]


def without_negative_assets(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[ASSET_COLUMN] >= 0]


def run_once(frame: pd.DataFrame, seed: int) -> dict:
    """Split, train, test and run the what-if experiment once, on one dataset."""
    train, test = split_dataset(frame, seed=seed)
    bundle = train_model(train, seed=seed)
    metrics = evaluate_model(bundle.pipeline, test)
    results = run_experiment(
        test,
        lambda rows: approval_scores(bundle.pipeline, rows),
        bundle.search_space,
    )
    summary = summarize(results)
    row = {"records": len(frame), "model": bundle.model_name}
    row.update({name: metrics[name] for name in METRIC_COLUMNS})
    row["eligible"] = 0 if results.empty else int(results["loan_id"].nunique())
    for strategy in Strategy:
        name = f"{strategy.value}_flip_pct"
        present = strategy.value in summary.index
        row[name] = summary.loc[strategy.value, "flip_rate_pct"] if present else float("nan")
    return row


def sensitivity(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Compare every result with and without the negative residential-asset records."""
    runs = {
        "all records": run_once(frame, seed),
        "without negative assets": run_once(without_negative_assets(frame), seed),
    }
    table = pd.DataFrame.from_dict(runs, orient="index", columns=RUN_COLUMNS)
    table.index.name = "dataset"
    return table


def seed_check(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """Repeat the whole pipeline with different seeds, so with different splits."""
    table = pd.DataFrame(
        [run_once(frame, seed) for seed in seeds],
        index=pd.Index(seeds, name="seed"),
        columns=RUN_COLUMNS,
    )
    return table


def spread(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, lowest and highest of each number across the seeds."""
    columns = [*METRIC_COLUMNS, *FLIP_COLUMNS]
    stats = table[columns].agg(["mean", "std", "min", "max"])
    stats.index.name = "across seeds"
    return stats
