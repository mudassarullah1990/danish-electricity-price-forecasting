import numpy as np
import pandas as pd

from danish_electricity_price_forecasting.train import (
    FEATURE_COLUMNS,
    train_model,
)


def test_catboost_training_and_prediction():
    """Train a small CatBoost model and verify its predictions."""

    rng = np.random.default_rng(42)
    n_samples = 240

    timestamps = pd.date_range(
        "2020-01-01",
        periods=n_samples,
        freq="h",
        tz="UTC",
    )

    data = pd.DataFrame(
        {
            "hour": timestamps.hour,
            "day_of_week": timestamps.day_of_week,
            "month": timestamps.month,
            "day_of_year": timestamps.day_of_year,
            "is_weekend": (timestamps.day_of_week >= 5).astype(int),
            "price_lag_48h": rng.uniform(10, 60, n_samples),
            "price_lag_72h": rng.uniform(10, 60, n_samples),
            "price_lag_168h": rng.uniform(10, 60, n_samples),
        }
    )

    data["target_price"] = (
        20
        + 0.4 * data["price_lag_48h"]
        + 0.2 * data["price_lag_72h"]
        + rng.normal(0, 1, n_samples)
    )

    train = data.iloc[:180].copy()
    validation = data.iloc[180:].copy()

    model = train_model(
        train,
        validation,
        iterations=20,
        learning_rate=0.1,
        depth=3,
        early_stopping_rounds=5,
        random_seed=42,
    )

    predictions = model.predict(validation[FEATURE_COLUMNS])

    assert len(predictions) == len(validation)
    assert np.isfinite(predictions).all()
    assert model.is_fitted()
