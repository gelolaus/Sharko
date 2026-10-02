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
