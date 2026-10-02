# Sharko

Sharko is a Python command-line app that trains a loan-approval classifier on `loan_approval_dataset.csv`. When an application is predicted Rejected, it searches for the closest loan amount, loan term, or both that flips the prediction to Approved.

Python 3.10 or newer.

## Install

```
python -m pip install -r requirements.txt
```

## Commands

Defaults: `--data loan_approval_dataset.csv`, `--model artifacts/model.joblib`, `--out outputs`, `--history artifacts/history.jsonl`.

Train a model and save the bundle. Prints 5-fold CV ROC-AUC for logistic regression, random forest, and gradient boosting, then held-out test metrics.

```
python -m sharko train
```

Recompute the saved model's test metrics.

```
python -m sharko evaluate
```

Search flips for every eligible held-out row (recorded Rejected and predicted Rejected). Prints the eligible count, a three-strategy summary, and the disclaimer.

```
python -m sharko experiment
```

Write summary tables and figures from the experiment results.

```
python -m sharko report
```

Score one application. With `--no-prompt`, pass every applicant field. Without it, Sharko asks for anything missing.

```
python -m sharko check --no-prompt --education Graduate --self-employed No --no-of-dependents 2 --income-annum 4100000 --cibil-score 417 --residential-assets-value 2700000 --commercial-assets-value 2200000 --luxury-assets-value 8800000 --bank-asset-value 3300000 --loan-amount 12200000 --loan-term 8
```

List logged check runs.

```
python -m sharko history
```

Run the tests.

```
python -m pytest
```

## Output locations

- `artifacts/model.joblib` - trained model bundle
- `artifacts/history.jsonl` - append-only log of check runs
- `outputs/experiment_results.csv` - one row per eligible application and strategy
- `outputs/summary.csv` - strategy summary
- `outputs/paired_comparison.csv` - paired comparisons of mutual successes
- `outputs/flip_rate_by_strategy.png`
- `outputs/distance_by_strategy.png`
- `outputs/configs_evaluated_by_strategy.png`
- `outputs/amount_original_vs_flip.png` - written when an amount change flips a prediction

`artifacts/` and `outputs/` are generated and git-ignored.

## Search rules

The split is 80/20, stratified on `loan_status`, seed 42. Search bounds, normalization, and model selection use the training partition only. Approved means the model approval probability is at least 0.5.

A search may change `loan_amount` and `loan_term`. Every other applicant field stays fixed. The three strategies are Amount-only, Term-only, and Combined. Candidates are ordered by normalized distance, then amount, then term. Results list all three. The smallest normalized distance can be labeled closest to your request.

## Ethics and limitations

Every user-facing result shows the original and modified amount and term, the model score, and the normalized distance, plus test metrics and this statement:

```
Model-generated what-if result. This is not a lender decision, a loan offer, or financial advice.
Limitations: the model reflects patterns in a public dataset that lacks economic variables (e.g. inflation, interest rates) and demographic detail; a missing flip means none was found within the searched amounts and terms, not that none exists.
```

Sharko reports the what-if result for a person to read. Filing an application stays with the applicant and the lender.

## Roadmap

Deferred after v1:

- Sensitivity analysis on the 28 negative-asset rows.
- History-based warm-start.

Retraining on history is rejected for v1. History is write-only and must not change a search or the trained model.
