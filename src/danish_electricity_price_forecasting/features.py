import pandas as pd


def create_features(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Create leakage-safe features for next-day hourly price forecasting."""
    target_column = f"{market}_price_day_ahead"

    if target_column not in df.columns:
        raise ValueError(f"Missing price column: {target_column}")

    result = df[["utc_timestamp", target_column]].copy()
    result = result.sort_values("utc_timestamp").reset_index(drop=True)

    timestamps = result["utc_timestamp"]

    # Calendar features known in advance.
    result["hour"] = timestamps.dt.hour
    result["day_of_week"] = timestamps.dt.dayofweek
    result["month"] = timestamps.dt.month
    result["day_of_year"] = timestamps.dt.dayofyear
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)

    # Historical prices from previous days.
    # Minimum lag of 48 hours is a conservative day-ahead assumption.
    result["price_lag_48h"] = result[target_column].shift(48)
    result["price_lag_72h"] = result[target_column].shift(72)
    result["price_lag_168h"] = result[target_column].shift(168)

    # Previous-week same-hour price as the baseline.
    result["baseline_price"] = result["price_lag_168h"]

    # The target is used for supervised training, not as a model input.
    result = result.rename(columns={target_column: "target_price"})

    feature_columns = [
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "price_lag_48h",
        "price_lag_72h",
        "price_lag_168h",
    ]

    result = result.dropna(subset=feature_columns + ["target_price"]).reset_index(
        drop=True
    )

    return result
