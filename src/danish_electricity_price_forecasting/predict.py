"""Generate day-ahead forecasts using Click, Hydra, CatBoost, and DVC."""

import subprocess
from datetime import timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from hydra import compose, initialize_config_dir

from danish_electricity_price_forecasting.data import load_data
from danish_electricity_price_forecasting.train import FEATURE_COLUMNS


def ensure_dvc_file(
    project_root: Path,
    file_path: Path,
    remote: str,
) -> Path:
    """Download a missing dataset or model from its DVC remote."""
    if file_path.is_file():
        return file_path

    dvc_file = file_path.with_name(file_path.name + ".dvc")

    if not dvc_file.is_file():
        raise FileNotFoundError(f"DVC tracking file not found: {dvc_file}")

    print(f"Downloading missing file from DVC remote '{remote}'...")

    subprocess.run(
        [
            "dvc",
            "pull",
            str(dvc_file.relative_to(project_root)),
            "-r",
            remote,
        ],
        cwd=project_root,
        check=True,
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"DVC download completed but file is missing: {file_path}"
        )

    print(f"File downloaded successfully: {file_path}")
    return file_path


def predict_day(forecast_date: str) -> pd.DataFrame:
    """Predict hourly electricity prices for the requested UTC date."""
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(config_name="config")

    data_path = project_root / cfg.data.raw_path
    models_dir = project_root / "outputs" / "models"
    output_dir = project_root / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_date = pd.Timestamp(forecast_date)

    if requested_date.tzinfo is not None:
        raise ValueError("Provide a date in YYYY-MM-DD format.")

    requested_date = requested_date.tz_localize("UTC").normalize()

    issue_time = (
        requested_date
        - timedelta(days=1)
        + timedelta(hours=cfg.forecast.issue_hour_utc)
    )

    forecast_hours = pd.date_range(
        start=requested_date,
        periods=cfg.forecast.horizon_hours,
        freq="h",
        tz="UTC",
    )

    # Retrieve the dataset through DVC if it is missing locally.
    data_path = ensure_dvc_file(
        project_root=project_root,
        file_path=data_path,
        remote="gdrive",
    )

    data = load_data(data_path)
    data = data.set_index("utc_timestamp").sort_index()

    results = pd.DataFrame({"utc_timestamp": forecast_hours})

    for market in cfg.markets:
        price_column = f"{market}_{cfg.data.target_suffix}"
        model_path = models_dir / f"{market}_catboost.cbm"

        # Retrieve the trained model through DVC if it is missing.
        model_path = ensure_dvc_file(
            project_root=project_root,
            file_path=model_path,
            remote="models",
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


@click.command()
@click.argument(
    "forecast_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
)
def main(forecast_date) -> None:
    """Generate day-ahead electricity price forecasts."""
    predict_day(forecast_date.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    main()
