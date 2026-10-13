# Sample outputs

These are real outputs from one full run of Sharko, so a reviewer can see the results without running anything. Run it yourself to get the same files. The pipeline is deterministic (seed 42), so the numbers match.

- Made on: Windows 11, Python 3.14.7, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, joblib 1.6.0
- Data: `loan_approval_dataset.csv` in the project folder (4,269 loans, INR)
- The example applicant is the built-in sample from `check --example`. It is not a real person.
- The model file and the history log are not included. `train` rebuilds the model, and the history log holds only your own checks.

## How to reproduce

Run these from the project folder, in this order.

```
python -m pip install -r requirements.txt
python -m sharko train
python -m sharko evaluate
python -m sharko experiment
python -m sharko report
python -m sharko scenarios
python -m sharko robustness
python -m sharko check --example
```

The files are written to `outputs/` and `Reports/`, which are git-ignored. This folder is a copy of them.

## What is here

| File | Made by | What it shows |
| --- | --- | --- |
| `terminal/train.txt` | `train` | the model comparison and the held-out scores |
| `terminal/evaluate.txt` | `evaluate` | the scores plus the Brier score and calibration table |
| `terminal/experiment.txt` | `experiment` | the 3 ways compared on all 312 eligible test loans |
| `terminal/report.txt` | `report` | which tables and charts were written |
| `terminal/scenarios.txt` | `scenarios` | four worked examples and the comparison by applicant profile |
| `terminal/robustness.txt` | `robustness` | results with and without the 28 negative-asset loans, and across 5 splits |
| `terminal/check_example.txt` | `check --example` | one full result as a user sees it |
| `outputs/experiment_results.csv` | `experiment` | one row per loan and way (936 rows) |
| `outputs/summary.csv` | `report` | one row per way, with all the averages |
| `outputs/paired_comparison.csv` | `report` | the ways compared on loans both could fix |
| `outputs/scenarios.csv` | `scenarios` | the comparison by CIBIL score, income and loan size |
| `outputs/robustness_sensitivity.csv` | `robustness` | with and without the negative-asset loans |
| `outputs/robustness_seeds.csv` | `robustness` | one row per random split |
| `outputs/*.png` | `report` | the paper's four charts |
| `reports/Experiment_Report.html` | `report` | all four charts on one page |
| `reports/Example_Applicant_Results.html` | `check --example` | the report a user gets for one application |

The two HTML files are single files. Open them in any browser.

Everything here is a model-generated what-if result. It is not a lender decision, a loan offer, or financial advice.
