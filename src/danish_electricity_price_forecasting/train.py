import pandas as pd
from catboost import CatBoostRegressor

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "price_lag_48h",
    "price_lag_72h",
    "price_lag_168h",
]

TARGET_COLUMN = "target_price"


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split observations chronologically using complete UTC days."""
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Training and validation ratios must sum to less than 1.")

    data = df.sort_values("utc_timestamp").copy()
    data["forecast_date"] = data["utc_timestamp"].dt.floor("D")

    # Retain only days with all 24 hourly target observations.
    complete_days = data.groupby("forecast_date").size()
    complete_days = complete_days[complete_days == 24].index

    data = data[data["forecast_date"].isin(complete_days)].copy()
    dates = data["forecast_date"].drop_duplicates().sort_values()

    n_days = len(dates)
    train_end = int(n_days * train_ratio)
    validation_end = train_end + int(n_days * validation_ratio)

    if train_end == 0 or validation_end <= train_end or validation_end >= n_days:
        raise ValueError("Not enough complete days for the requested split.")

    train_dates = dates.iloc[:train_end]
    validation_dates = dates.iloc[train_end:validation_end]
    test_dates = dates.iloc[validation_end:]

    train = data[data["forecast_date"].isin(train_dates)].copy()
    validation = data[data["forecast_date"].isin(validation_dates)].copy()
    test = data[data["forecast_date"].isin(test_dates)].copy()

    for subset in (train, validation, test):
        subset.drop(columns="forecast_date", inplace=True)
        subset.reset_index(drop=True, inplace=True)

    return train, validation, test


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 50,
    random_seed: int = 42,
) -> CatBoostRegressor:
    """Train a CatBoost model using chronological validation data."""
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
        eval_set=(
            validation[FEATURE_COLUMNS],
            validation[TARGET_COLUMN],
        ),
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
    )

    return model
