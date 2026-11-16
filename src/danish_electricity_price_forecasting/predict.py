"""Generate 24-hour day-ahead forecasts for Danish electricity markets."""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from danish_electricity_price_forecasting.data import load_data
from danish_electricity_price_forecasting.train import FEATURE_COLUMNS

MARKETS = ("DK_1", "DK_2")
ISSUE_HOUR_UTC = 12
HORIZON_HOURS = 24


def predict_day(forecast_date: str) -> pd.DataFrame:
    """Predict all 24 UTC hours of the requested date."""
    project_root = Path(__file__).resolve().parents[2]

    data_path = project_root / "data/raw/time_series_60min_singleindex.csv"
    models_dir = project_root / "outputs/models"
    output_dir = project_root / "outputs/predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_date = pd.Timestamp(forecast_date)

    if requested_date.tzinfo is not None:
        raise ValueError("Provide a date in YYYY-MM-DD format.")

    requested_date = requested_date.tz_localize("UTC").normalize()

    issue_time = requested_date - timedelta(days=1) + timedelta(hours=ISSUE_HOUR_UTC)

    forecast_hours = pd.date_range(
        start=requested_date,
        periods=HORIZON_HOURS,
        freq="h",
        tz="UTC",
    )

    data = load_data(data_path)
    data = data.set_index("utc_timestamp").sort_index()

    results = pd.DataFrame({"utc_timestamp": forecast_hours})

    for market in MARKETS:
        price_column = f"{market}_price_day_ahead"
        model_path = models_dir / f"{market}_catboost.cbm"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run the training pipeline first."
            )

        features = pd.DataFrame(
            {
                "hour": forecast_hours.hour,
                "day_of_week": forecast_hours.day_of_week,
                "month": forecast_hours.month,
                "day_of_year": forecast_hours.day_of_year,
                "is_weekend": (forecast_hours.day_of_week >= 5).astype(int),
            }
        )

        for lag in (48, 72, 168):
            history_hours = forecast_hours - pd.Timedelta(hours=lag)

            if (history_hours > issue_time).any():
                raise ValueError(
                    f"Lag {lag}h requires prices unavailable "
                    "at the forecast issuance time."
                )

            historical_prices = data[price_column].reindex(history_hours)

            if historical_prices.isna().any():
                raise ValueError(
                    f"Missing historical {market} prices required for the {lag}h lag."
                )

            features[f"price_lag_{lag}h"] = historical_prices.to_numpy()

        model = CatBoostRegressor()
        model.load_model(str(model_path))

        predictions = model.predict(features[FEATURE_COLUMNS])

        if not np.isfinite(predictions).all():
            raise ValueError(f"Invalid predictions generated for {market}.")

        results[f"{market}_predicted_price"] = predictions

    output_path = output_dir / f"day_ahead_forecast_{requested_date:%Y-%m-%d}.csv"

    results.to_csv(output_path, index=False)

    print("\n===== DAY-AHEAD ELECTRICITY PRICE FORECAST =====")
    print(f"Forecast date (UTC): {requested_date:%Y-%m-%d}")
    print(f"Issuance time (UTC): {issue_time}")
    print(results.to_string(index=False))
    print(f"\nForecast saved: {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 24-hour Danish electricity price forecasts."
    )
    parser.add_argument(
        "date",
        help="Forecast date in YYYY-MM-DD format",
    )
    args = parser.parse_args()

    predict_day(args.date)


if __name__ == "__main__":
    main()
