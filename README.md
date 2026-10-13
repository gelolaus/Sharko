# Sharko

Sharko is a small command-line app that answers one question about a loan:

> "If this application would be **rejected**, what is the **smallest change** to the loan amount or loan term that would get it **approved**?"

It learns from a public dataset of past loan decisions (`loan_approval_dataset.csv`, an Indian lending setting). **All amounts are in Indian Rupees (INR).** It is a course project for MODESIM, and its results are not a lender decision or financial advice.

## How it works

1. **Learn.** Sharko trains a model on past loan decisions. The model gives any application a *chance of approval* from 0% to 100%. 50% or more counts as approved.
2. **Check.** You enter one application. If the chance is 50% or more, you are done.
3. **Search.** If it is below 50%, Sharko tries other loan amounts and terms, nearest to your request first. Everything else about the applicant stays the same. It stops at the first option the model would approve.
4. **Compare.** Sharko does the search three ways (change the amount only, the term only, or both) and shows all three side by side.

Example from a real run: an application with a 0.3% chance of approval becomes 54.1% if the loan is changed from 12,200,000 INR over 8 years to 14,600,000 INR over 4 years. Changing only the amount or only the term found nothing.

## Install

You need Python 3.10 or newer.

```
python -m pip install -r requirements.txt
```

## Quick start

Run these from the project folder.

```
python -m sharko train
python -m sharko check --example
```

1. `train` teaches Sharko from the loan data. Run it once.
2. `check --example` shows a full worked example and saves a report in the `Reports` folder. No typing needed.

Then try your own application. Sharko asks for your first and last name, then 11 short questions, and shows the allowed answers under each one. Press Ctrl+C to quit.

```
python -m sharko check
```

Stuck? `python -m sharko tutorial` prints a step-by-step guide, and `python -m sharko` on its own shows the quick start.

## Your report

Every `check` saves an HTML report on its own, so there is nothing extra to run. Look in the `Reports` folder inside the Sharko folder for `FirstName_LastName_Results.html`. The terminal prints the folder and file name when it is saved, and asks if you want to open it. Add `--open` to open it automatically.

The report is one file you can open in any browser or send to a teammate. It holds:

- all the answers you submitted
- the result and the smallest change that would be approved
- how Sharko found it, in 4 plain steps
- the three ways compared
- every try the search made, with a small chart of the approval chance at each try
- the model's quality scores, the limits and the disclaimer

Running `check` again with the same name saves a new file ending in `_2`, `_3` and so on, so nothing is overwritten. The report contains names and financial details, so `Reports/` is git-ignored. Names are only used for the report. They are not given to the model and not written to the history log.

## Reading the result

The result is built to be read from the top, and you can stop early.

| Section | What it tells you |
| --- | --- |
| RESULT | Approved or not, with the model's chance of approval |
| SMALLEST CHANGE THAT WOULD BE APPROVED | The closest loan amount and term that works |
| HOW SHARKO FOUND THIS | The 4 steps above, with this run's numbers |
| THE 3 WAYS COMPARED | Amount only, term only, both |
| IMPORTANT | What this result is and is not |
| TECHNICAL DETAILS | Exact numbers for analysts. Safe to skip. |

A bigger loan can come out as "approved". That only reflects patterns in the past data. It does not mean a bigger loan is safer.

## See the charts

Run the experiment once, then open the dashboard. Both commands work from the terminal and the charts open in your browser.

```
python -m sharko experiment
python -m sharko report --open
```

`report` writes four charts and one page, `Reports/Experiment_Report.html`, that holds all of them with a plain-language caption each. You can send that single file to a teammate. The charts follow the paper's proposed visualizations:

- Prediction-flip rate by strategy (bar chart)
- Minimum normalized distance by strategy (box plot)
- Configurations evaluated by strategy (box plot, log scale; shows how many loans found an option and how many tried every option)
- Original vs. prediction-flipping loan amount (scatter plot)

Without `--open`, `report` only writes the files.

## Check how steady the results are

`python -m sharko robustness` answers two questions a reviewer is likely to ask. It prints both tables and saves them as `outputs/robustness_sensitivity.csv` and `outputs/robustness_seeds.csv`.

1. **Do the 28 loans with a negative residential asset value matter?** It trains, tests and runs all 3 searches twice, once on every loan and once without those 28, and puts the scores side by side.
2. **Do the results depend on one lucky split?** It repeats the whole run with 5 different random splits (seeds 42 to 46) and reports the mean, spread and range. Use `--seeds 10` for more.

In a real run on the full dataset (4,269 loans):

| Run | Accuracy | Amount only | Term only | Both |
| --- | --- | --- | --- | --- |
| All loans | 0.982 | 7.1% | 47.8% | 100% |
| Without the 28 negative-asset loans | 0.982 | 8.9% | 48.9% | 100% |
| 5 splits (seeds 42 to 46), mean | 0.980 | 8.1% | 50.4% | 100% |

Removing the 28 loans changes almost nothing, and the order of the three ways is the same on every split (both > term only > amount only). The columns are the share of eligible test loans where that way found an approved option. Run it yourself for the exact numbers on your machine.

## Compare applicant profiles

`python -m sharko scenarios` shows how the three ways behave for different kinds of applicant. It takes the same eligible test loans as `experiment` and prints:

- four worked examples taken from those loans: the lowest, middle and highest CIBIL score, and the largest loan for the income
- the three ways compared for the lowest, middle and highest third of CIBIL score, annual income and loan amount relative to income

It saves the table to `outputs/scenarios.csv`. In a real run the CIBIL score barely changed the results and income changed them a little, but the loan size changed them a lot: changing only the term found an approved option for 13.5% of the loans with the smallest loan relative to income, and for 92.3% of those with the largest. Changing both found one for every loan in every group.

## How far to trust the percentages

`python -m sharko evaluate` also checks whether the model's chance of approval means what it says. It prints the Brier score (0 is perfect) and groups the test loans by the chance the model gave them, next to the share that were truly Approved.

In a real run the Brier score was 0.014 and the average gap was 1.3 percentage points. The model is reliable at the ends: loans it scored under 10% or over 90% were almost always rejected or approved as predicted. The middle is thin. Only about 50 test loans scored between 10% and 90%, and the 80% to 90% group was truly approved less than half the time (9 loans). Sharko stops at the first option with 50% or more, so a suggested change usually lands in that middle zone. Read a suggested "52% chance" as a weaker signal than a "99% chance".

## Commands

| Command | What it does |
| --- | --- |
| `python -m sharko train` | Teach Sharko from the loan data (run once) |
| `python -m sharko check` | Check one application and save your report in `Reports` |
| `python -m sharko check --example` | Check a built-in sample application (also saves a report) |
| `python -m sharko history` | List your earlier checks |
| `python -m sharko tutorial` | Step-by-step guide |
| `python -m sharko evaluate` | Show how accurate the saved model is and whether its percentages can be trusted |
| `python -m sharko experiment` | For analysts: run the search on many test loans |
| `python -m sharko scenarios` | For analysts: compare the three ways for different kinds of applicant |
| `python -m sharko robustness` | For analysts: check the results with and without the negative-asset loans and across random splits |
| `python -m sharko report --open` | For analysts: draw the charts and open the dashboard in your browser |

Every command accepts `--help`. To skip the questions, pass every answer as a flag with `check --no-prompt`. See `python -m sharko check --help` for the flag names.

Run the tests with `python -m pytest`.

## How long things take

Measured on Windows 11 with Python 3.14. A slower machine, or the first run, takes longer. The install time depends on your connection.

| Command | About |
| --- | --- |
| `python -m pip install -r requirements.txt` | 1 to 3 minutes |
| `python -m sharko train` | 10 seconds |
| `python -m sharko evaluate` | 5 seconds |
| `python -m sharko check --example` | 6 seconds |
| `python -m sharko experiment` | 25 seconds |
| `python -m sharko scenarios` | 25 seconds |
| `python -m sharko report --open` | 6 seconds |
| `python -m sharko robustness` | 3 minutes (6 full runs) |
| `python -m pytest` | 2 to 3 minutes |

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `Model not found ... Run: python -m sharko train` | Run `python -m sharko train` first. It saves the model in `artifacts/`. |
| `Dataset not found` | Run the command from the Sharko folder, or pass `--data path/to/loan_approval_dataset.csv`. |
| `Experiment results not found` | Run `python -m sharko experiment` before `report`. |
| `No module named 'sklearn'` (or `pandas`, `matplotlib`) | Run `python -m pip install -r requirements.txt` with the same `python` you use for Sharko. |
| `python` is too old or not found | Sharko needs Python 3.10 or newer. Check with `python --version`. On macOS or Linux try `python3`. On Windows try `py` in place of `python`. |
| The report did not open in a browser | Open the file in `Reports/` by double-clicking it. |
| A prompt says an answer is not allowed | Read the options printed under the question. Amounts are in steps of 100,000 INR and terms are even years from 2 to 20. |
| `experiment`, `scenarios` or `robustness` seems stuck | It is working. `experiment` prints progress. `scenarios` takes under a minute and `robustness` takes a few minutes. |
| `train` needs to be re-run | Run it again any time. It overwrites `artifacts/model.joblib`. |


## Files Sharko writes

These folders are generated and git-ignored.

- `artifacts/model.joblib` - the trained model
- `artifacts/history.jsonl` - a log of your checks (write-only, it never changes a result)
- `Reports/` - your reports (`FirstName_LastName_Results.html`) and the charts page (`Experiment_Report.html`)
- `outputs/` - `experiment_results.csv`, `summary.csv`, `paired_comparison.csv`, `scenarios.csv`, `robustness_sensitivity.csv`, `robustness_seeds.csv` and the charts (`.png`)

A copy of one full run (terminal output, CSVs, charts and example reports) is kept in `samples/`, so you can see the results without running anything. See `samples/README.md`.

## For analysts

- The data is split 80/20 (stratified on `loan_status`, seed 42). Search limits, scaling and model choice use the 80% training part only. The model is picked by cross-validated ROC-AUC from logistic regression, random forest and gradient boosting.
- Only `loan_amount` and `loan_term` change during a search. Every other field stays fixed.
- Options are ordered by normalized distance, then amount, then term. Ties are decided on the exact integer numerator, never on floats.
- Sharko never names one "best" way with a weighted score. It reports the three ways side by side.
- Every result shows the original and new amount and term, the model score and the distance, plus the disclaimer.

## Documentation

`Sharko_Documentation.pdf` is the full guide: how it works, every command, how to read the results, how steady and reliable they are, and how the code is organized. Its source is `documentation/sharko_documentation.html`. To rebuild the PDF, print that file to PDF from Edge or Chrome with no headers or footers.

## Ethics and limitations

```
Model-generated what-if result. This is not a lender decision,
a loan offer, or financial advice. A high chance of approval is
an estimate, not a guarantee that a real lender would approve.
Limitations: the model reflects patterns in a public dataset that
lacks economic variables (e.g. inflation, interest rates) and
demographic detail. If no approved option is found, that means none
was found within the searched amounts and terms, not that none exists.
Scores can change with a different data split; run
python -m sharko robustness to see by how much.
```

The percentages are most reliable near 0% and 100% and least reliable in the middle, which is where a suggested change usually lands (see "How far to trust the percentages"). Sharko makes no claim that the model is fair across groups, because the data has little demographic detail.

Sharko reports a what-if result for a person to read. Filing an application stays with the applicant and the lender.

## Roadmap

Deferred after v1: history-based warm-start. Retraining on history is rejected for v1. The sensitivity analysis on the 28 negative-asset rows and the check across random splits are done: see `python -m sharko robustness`.
