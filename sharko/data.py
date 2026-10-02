from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.model_selection import train_test_split

from sharko.config import AMOUNT_STEP, SEED


def load_dataset(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True, encoding="utf-8")
    frame.columns = frame.columns.str.strip()
    for column in frame.columns:
        if is_object_dtype(frame[column].dtype) or is_string_dtype(
            frame[column].dtype
        ):
            frame[column] = frame[column].str.strip()
    return frame


def split_dataset(
    df: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["loan_status"],
        random_state=seed,
    )
    return train, test


@dataclass(frozen=True)
class SearchSpace:
    amount_min: int
    amount_max: int
    amount_step: int
    terms: tuple[int, ...]

    @property
    def amount_range(self) -> int:
        return self.amount_max - self.amount_min

    @property
    def term_range(self) -> int:
        return max(self.terms) - min(self.terms)


def derive_search_space(train: pd.DataFrame) -> SearchSpace:
    return SearchSpace(
        amount_min=int(train["loan_amount"].min()),
        amount_max=int(train["loan_amount"].max()),
        amount_step=AMOUNT_STEP,
        terms=tuple(int(term) for term in sorted(train["loan_term"].unique())),
    )
