import pytest

from sharko.data import split_dataset
from sharko.model import (
    approval_scores,
    evaluate_model,
    load_bundle,
    predicted_status,
    save_bundle,
    train_model,
)


@pytest.fixture
def train_test(synthetic_df):
    return split_dataset(synthetic_df)


def test_train_model_reports_cv_for_all_candidates(train_test):
    b = train_model(train_test[0])
    assert set(b.cv_auc) == {"logistic_regression", "random_forest", "gradient_boosting"}
    assert b.model_name in b.cv_auc and all(0.0 <= v <= 1.0 for v in b.cv_auc.values())
    assert b.test_metrics is None


def test_scores_are_row_aligned_probabilities(train_test):
    b = train_model(train_test[0])
    s = approval_scores(b.pipeline, train_test[1])
    assert s.shape == (len(train_test[1]),) and ((s >= 0) & (s <= 1)).all()


def test_model_learns_synthetic_rule(train_test):
    m = evaluate_model(train_model(train_test[0]).pipeline, train_test[1])
    assert m["accuracy"] >= 0.95 and m["roc_auc"] >= 0.97
    cm = m["confusion_matrix"]
    assert sum(map(sum, cm)) == len(train_test[1])


def test_training_is_deterministic(train_test):
    a, b = train_model(train_test[0]), train_model(train_test[0])
    assert (approval_scores(a.pipeline, train_test[1]) == approval_scores(b.pipeline, train_test[1])).all()


def test_predicted_status_threshold():
    assert predicted_status(0.5) == "Approved" and predicted_status(0.4999) == "Rejected"


def test_bundle_roundtrip(train_test, tmp_path):
    b = train_model(train_test[0])
    save_bundle(b, tmp_path / "sub" / "m.joblib")
    r = load_bundle(tmp_path / "sub" / "m.joblib")
    assert r.search_space == b.search_space and r.model_name == b.model_name
    assert (approval_scores(r.pipeline, train_test[1]) == approval_scores(b.pipeline, train_test[1])).all()
