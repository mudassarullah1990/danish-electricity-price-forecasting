"""Training pipeline with Hydra, CatBoost, and MLflow tracking."""

import subprocess
from pathlib import Path

import mlflow
from hydra import compose, initialize_config_dir
from sklearn.metrics import mean_absolute_error, mean_squared_error

from danish_electricity_price_forecasting.data import load_data
from danish_electricity_price_forecasting.evaluate import create_evaluation_plots
from danish_electricity_price_forecasting.features import create_features
from danish_electricity_price_forecasting.train import (
    FEATURE_COLUMNS,
    split_data,
    train_model,
)


def get_git_commit(project_root: Path) -> str:
    """Return the current Git commit identifier."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def calculate_metrics(actual, predicted) -> dict[str, float]:
    """Calculate regression error metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def main() -> None:
    """Train and track Danish electricity price forecasting models."""
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"

    # Load Hydra configuration
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(config_name="config")

    # Configure MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Get Git commit and load dataset
    git_commit = get_git_commit(project_root)
    data = load_data(project_root / cfg.data.raw_path)

    print("===== DANISH ELECTRICITY PRICE FORECASTING =====")

    for market in cfg.markets:
        print(f"\nProcessing market: {market}")

        # Create features
        features = create_features(data, market)

        # Chronological train/validation/test split
        train, validation, _test = split_data(
            features,
            train_ratio=cfg.split.train_ratio,
            validation_ratio=cfg.split.validation_ratio,
        )

        # Create separate MLflow run for each market
        with mlflow.start_run(run_name=f"{market}-catboost"):
            # Log experiment metadata
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("market", market)
            mlflow.set_tag("model_type", cfg.model.name)

            # Log model parameters
            mlflow.log_params(
                {
                    "market": market,
                    "model": cfg.model.name,
                    "iterations": cfg.model.iterations,
                    "learning_rate": cfg.model.learning_rate,
                    "depth": cfg.model.depth,
                    "early_stopping_rounds": cfg.model.early_stopping_rounds,
                    "random_seed": cfg.model.random_seed,
                    "train_ratio": cfg.split.train_ratio,
                    "validation_ratio": cfg.split.validation_ratio,
                }
            )

            # Train CatBoost model
            model = train_model(
                train,
                validation,
                iterations=cfg.model.iterations,
                learning_rate=cfg.model.learning_rate,
                depth=cfg.model.depth,
                early_stopping_rounds=cfg.model.early_stopping_rounds,
                random_seed=cfg.model.random_seed,
            )

            # Generate validation predictions
            predictions = model.predict(validation[FEATURE_COLUMNS])

            # Calculate model performance
            metrics = calculate_metrics(
                validation["target_price"],
                predictions,
            )

            # Calculate previous-week baseline performance
            baseline_metrics = calculate_metrics(
                validation["target_price"],
                validation["baseline_price"],
            )

            # Log evaluation metrics
            mlflow.log_metrics(
                {
                    "validation_mae": metrics["mae"],
                    "validation_rmse": metrics["rmse"],
                    "baseline_mae": baseline_metrics["mae"],
                    "baseline_rmse": baseline_metrics["rmse"],
                    "best_iteration": model.get_best_iteration(),
                }
            )

            # Generate three evaluation plots
            plot_paths = create_evaluation_plots(
                model=model,
                validation=validation,
                predictions=predictions,
                feature_columns=FEATURE_COLUMNS,
                market=market,
                output_dir=project_root / "outputs" / "plots",
            )

            # Upload plots to MLflow
            for plot_path in plot_paths:
                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="plots",
                )

            print(f"\nMarket: {market}")
            print(f"Training observations: {len(train):,}")
            print(f"Validation observations: {len(validation):,}")
            print(f"MAE: {metrics['mae']:.4f} EUR/MWh")
            print(f"RMSE: {metrics['rmse']:.4f} EUR/MWh")
            print(f"Baseline MAE: {baseline_metrics['mae']:.4f} EUR/MWh")
            print(f"Baseline RMSE: {baseline_metrics['rmse']:.4f} EUR/MWh")
            print(f"Best iteration: {model.get_best_iteration()}")
            print(f"Evaluation plots logged: {len(plot_paths)}")
            print("MLflow run logged successfully.")

    print("\nTraining and MLflow tracking completed successfully!")


if __name__ == "__main__":
    main()
