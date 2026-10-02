from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from tests.synthetic import make_synthetic_df, write_synthetic_csv


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    return write_synthetic_csv(tmp_path / "loans.csv")


@dataclass
class Trained:
    csv: Path
    model: Path
    out: Path


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    from sharko.cli import main

    d = tmp_path_factory.mktemp("trained")
    t = Trained(write_synthetic_csv(d / "loans.csv"), d / "model.joblib", d / "out")
    assert main(["train", "--data", str(t.csv), "--model", str(t.model)]) == 0
    return t


@pytest.fixture(scope="session")
def trained_experiment(trained):
    from sharko.cli import main

    assert (
        main(
            [
                "experiment",
                "--data",
                str(trained.csv),
                "--model",
                str(trained.model),
                "--out",
                str(trained.out),
            ]
        )
        == 0
    )
    return trained
