# Day-Ahead Electricity Price Forecasting for Danish Markets

An end-to-end MLOps project for forecasting hourly day-ahead electricity prices in Denmark's two bidding zones, DK_1 and DK_2, using classical machine learning.

The project uses CatBoost regression and historical electricity prices, with reproducible data versioning, experiment tracking, configuration management, automated testing, and continuous integration.

## 1. Project Overview

Electricity prices vary throughout the day due to changes in demand, renewable generation, and market conditions. Accurate day-ahead price forecasts can support electricity trading, energy storage scheduling, and power system operation.

This project predicts 24 hourly electricity prices for the following UTC day in two Danish bidding zones:

- DK_1: Western Denmark
- DK_2: Eastern Denmark

The main objective is to compare CatBoost regression with a simple previous-week price baseline.

## 2. Dataset

**Source:** Open Power System Data (OPSD)

**Release:** 2020-10-06

**Dataset:** Hourly time series, single-index format

**Download:**
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv

The source dataset contains approximately 50,401 hourly observations and includes electricity prices and other power system variables.

The project uses the hourly day-ahead price columns:

- `DK_1_price_day_ahead`
- `DK_2_price_day_ahead`

Data is stored in:

`data/raw/time_series_60min_singleindex.csv`

The raw dataset is versioned using DVC, with Google Drive configured as remote storage.

## 3. Machine Learning Approach

### Baseline

The baseline predicts the electricity price using the price observed at the same hour seven days earlier (168-hour lag).

### Main Model

CatBoostRegressor is used to predict hourly electricity prices.

Features include:

- Hour of day
- Day of week
- Month
- Day of year
- Weekend indicator
- Electricity price lagged by 48 hours
- Electricity price lagged by 72 hours
- Electricity price lagged by 168 hours

Separate CatBoost models are trained for DK_1 and DK_2.

### Data Splitting

The dataset is split chronologically, preserving complete UTC days:

- Training: 70%
- Validation: 15%
- Testing: 15%

The validation set is used for model selection and early stopping. The held-out test set is used for final evaluation.

## 4. Project Structure

```text
danish-electricity-price-forecasting/
├── .github/
│   └── workflows/
│       └── ci.yml
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   └── opsd.yaml
│   └── model/
│       └── catboost.yaml
├── data/
│   └── raw/
├── outputs/
│   ├── models/
│   ├── plots/
│   ├── predictions/
│   └── results/
├── src/
│   └── danish_electricity_price_forecasting/
│       ├── __init__.py
│       ├── commands.py
│       ├── data.py
│       ├── evaluate.py
│       ├── features.py
│       ├── predict.py
│       └── train.py
├── tests/
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

## 5. Technology Stack

| Component                | Technology             |
| ------------------------ | ---------------------- |
| Programming language     | Python 3.12            |
| Package management       | uv                     |
| Data processing          | pandas, NumPy          |
| Machine learning         | CatBoost, scikit-learn |
| Configuration management | Hydra                  |
| Data versioning          | DVC                    |
| Remote data storage      | Google Drive           |
| Experiment tracking      | MLflow                 |
| Code quality             | Ruff, pre-commit       |
| Automated testing        | pytest                 |
| Continuous integration   | GitHub Actions         |
| Version control          | Git and GitHub         |

## 6. Setup

Clone the repository:

```bash
git clone https://github.com/mudassarullah1990/danish-electricity-price-forecasting.git
cd danish-electricity-price-forecasting
```

Install project dependencies:

```bash
uv sync --locked --dev
```

Restore the versioned dataset:

```bash
uv run dvc pull
```

**Note:** Access to the configured Google Drive DVC remote and its authentication credentials is required. The raw data can alternatively be downloaded from the public OPSD source above.

## 7. Train and Experiment Tracking

Start the MLflow tracking server in a separate terminal:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Run the training and final evaluation pipeline:

```bash
uv run python -m danish_electricity_price_forecasting.commands
```

The pipeline:

1. Loads the Hydra configuration and OPSD dataset.
2. Generates features for DK_1 and DK_2.
3. Splits observations chronologically.
4. Trains separate CatBoost models.
5. Evaluates validation and test performance.
6. Compares CatBoost against the previous-week baseline.
7. Logs parameters, metrics, plots, and models to MLflow.
8. Exports the final test results.

MLflow UI:

http://127.0.0.1:8080

## 8. Final Test Results

The final held-out test evaluation produced the following results.

All errors are in EUR/MWh; lower values are better.

| Metric           |    DK_1 |    DK_2 |
| ---------------- | ------: | ------: |
| CatBoost MAE     |  9.3480 |  9.2518 |
| Baseline MAE     | 10.4252 | 10.2843 |
| CatBoost RMSE    | 12.6704 | 12.8311 |
| Baseline RMSE    | 14.5724 | 14.7906 |
| MAE improvement  |  10.33% |  10.04% |
| RMSE improvement |  13.05% |  13.25% |

CatBoost outperformed the previous-week baseline on both Danish markets.

However, test errors were higher than validation errors, indicating that forecasting performance varied across the chronological evaluation periods.

The test set was not used for model tuning.

The detailed results are exported to:

`outputs/results/final_test_metrics.csv`

## 9. Day-Ahead Price Prediction

The project includes a command-line inference module that loads the trained CatBoost models and generates 24 hourly predictions for a specified historical date.

Example:

```bash
uv run python -m danish_electricity_price_forecasting.predict 2020-09-29
```

The example produces:

- 24 hourly forecasts for DK_1
- 24 hourly forecasts for DK_2
- UTC timestamps
- Electricity price predictions in EUR/MWh

The output is saved to:

`outputs/predictions/day_ahead_forecast_2020-09-29.csv`

The forecast is issued at 12:00 UTC on the preceding day.

The inference implementation checks that historical lagged prices are available before the configured issuance time.

This is a historical forecasting demonstration, not a live market forecasting service.

## 10. Testing and Code Quality

Run automated tests:

```bash
uv run pytest tests/ -v
```

The project currently has 19 passing automated tests covering data processing, feature engineering, model training, evaluation, and metrics.

Run pre-commit checks:

```bash
uv run pre-commit run --all-files
```

Ruff is used for linting and code formatting.

## 11. Continuous Integration

GitHub Actions automatically runs on pushes and pull requests targeting `main`.

The workflow:

1. Checks out the repository.
2. Installs uv and Python 3.12.
3. Installs locked project dependencies.
4. Runs automated pytest tests.
5. Runs pre-commit code quality checks.

The CI pipeline has completed successfully.

## 12. MLOps Reproducibility

The project implements:

- Git for source code version control
- DVC for dataset versioning
- Google Drive for remote dataset storage
- Hydra for configuration management
- MLflow for experiment tracking and model artifacts
- uv for reproducible dependency installation
- pytest for automated testing
- pre-commit for code quality enforcement
- GitHub Actions for continuous integration

The training pipeline records the Git commit identifier in MLflow to associate experiments with source code versions.

### Automatic DVC Artifact Retrieval

The project configures two DVC remotes:

- `gdrive`: versioned OPSD dataset
- `models`: trained CatBoost model files

Training automatically attempts to restore the dataset through DVC when the raw CSV is missing. Prediction automatically attempts to restore missing dataset and model files.

Both retrieval paths have been tested successfully, with restored files verified against the originals using SHA-256 checksums.

**Access requirements:** Google Drive DVC downloads require appropriate access permissions and authentication. Credentials are not included in the repository. Users without access can obtain the raw dataset from the public OPSD source linked in Section 2; trained models require access to the model remote or local retraining.

## 13. Limitations and Future Improvements

- The project uses historical OPSD data ending in 2020.
- The current implementation does not retrieve live market prices.
- The forecasting demonstration uses 24 UTC hours; Danish market delivery days can contain 23 or 25 local hours during daylight saving time transitions.
- All feature availability assumptions should be independently audited against the operational day-ahead market publication schedule before deployment.
- The current CatBoost model is trained on historical price and calendar features rather than future weather, demand, or renewable generation forecasts.
- Further research could investigate additional market features, time-series cross-validation, and model performance under changing electricity market conditions.

## 14. Conclusion

This project demonstrates an end-to-end classical machine learning and MLOps workflow for Danish day-ahead electricity price forecasting.

CatBoost achieved lower MAE and RMSE than the previous-week baseline in both DK_1 and DK_2 on the held-out test set.

The project also demonstrates reproducible data management, experiment tracking, automated tests, continuous integration, and a working historical 24-hour prediction command.

## Author

Mudassar Ullah

MSc Artificial Intelligence — MIPT

MLOps Course Project, Fall 2026
