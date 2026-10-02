from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sharko.config import (
    APPROVAL_THRESHOLD,
    APPROVED,
    REJECTED,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
)
from sharko.data import SearchSpace, derive_search_space


@dataclass
class ModelBundle:
    pipeline: Pipeline
    search_space: SearchSpace
    model_name: str
    cv_auc: dict[str, float]
    test_metrics: dict | None
    seed: int


def _encode_target(frame: pd.DataFrame) -> np.ndarray:
    return (frame[TARGET] == APPROVED).astype(int).to_numpy()


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURES]


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
    )


def _candidate_pipelines(seed: int) -> list[tuple[str, Pipeline]]:
    preprocessor = _make_preprocessor()
    return [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("preprocess", _make_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting",
            Pipeline(
                [
                    ("preprocess", _make_preprocessor()),
                    (
                        "model",
                        GradientBoostingClassifier(random_state=seed),
                    ),
                ]
            ),
        ),
    ]


def train_model(train: pd.DataFrame, seed: int = SEED) -> ModelBundle:
    x_train = _feature_matrix(train)
    y_train = _encode_target(train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    cv_auc: dict[str, float] = {}
    best_name = ""
    best_score = float("-inf")
    best_pipeline: Pipeline | None = None

    for name, pipeline in _candidate_pipelines(seed):
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
        )
        mean_score = float(scores.mean())
        cv_auc[name] = mean_score
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None
    best_pipeline.fit(x_train, y_train)

    return ModelBundle(
        pipeline=best_pipeline,
        search_space=derive_search_space(train),
        model_name=best_name,
        cv_auc=cv_auc,
        test_metrics=None,
        seed=seed,
    )


def evaluate_model(pipeline: Pipeline, test: pd.DataFrame) -> dict:
    x_test = _feature_matrix(test)
    y_test = _encode_target(test)
    scores = approval_scores(pipeline, test)
    y_pred = (scores >= APPROVAL_THRESHOLD).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "confusion_matrix": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
    }


def approval_scores(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(_feature_matrix(frame))[:, 1]


def predicted_status(score: float) -> str:
    if score >= APPROVAL_THRESHOLD:
        return APPROVED
    return REJECTED


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)
