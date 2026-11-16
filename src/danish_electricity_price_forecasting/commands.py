"""Training pipeline with Hydra, CatBoost, DVC, and MLflow tracking."""

import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen

import mlflow
import pandas as pd
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
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def download_data(project_root: Path, raw_path: str, source_url: str) -> Path:
    """Restore missing data using DVC, with a public OPSD fallback."""
    data_path = project_root / raw_path

    if data_path.is_file():
        print(f"Dataset already available: {data_path}")
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    dvc_file = data_path.with_name(data_path.name + ".dvc")

    print("Dataset missing. Attempting DVC retrieval...")

    try:
        subprocess.run(
            [
                "dvc",
                "pull",
                str(dvc_file.relative_to(project_root)),
                "-r",
                "gdrive",
            ],
            cwd=project_root,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"DVC retrieval unavailable: {exc}")

    if not data_path.is_file():
        print("Downloading dataset from public OPSD source...")
        temporary_path = data_path.with_name(data_path.name + ".part")

        try:
            with (
                urlopen(source_url, timeout=60) as response,
                temporary_path.open("wb") as destination,
            ):
                shutil.copyfileobj(response, destination)

            temporary_path.replace(data_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset retrieval completed but file was not found: {data_path}"
        )

    print(f"Dataset available: {data_path}")
    return data_path


def main() -> None:
    """Train, evaluate, save, and track Danish electricity price models."""
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(config_name="config")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    git_commit = get_git_commit(project_root)

    # Retrieve the dataset through DVC or the public OPSD fallback.
    data_path = download_data(
        project_root,
        cfg.data.raw_path,
        cfg.data.source_url,
    )
    data = load_data(data_path)

    models_dir = project_root / "outputs" / "models"
    results_dir = project_root / "outputs" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    test_results = []

    print("===== DANISH ELECTRICITY PRICE FORECASTING =====")

    for market in cfg.markets:
        print(f"\nProcessing market: {market}")

        features = create_features(data, market)

        train, validation, test = split_data(
            features,
            train_ratio=cfg.split.train_ratio,
            validation_ratio=cfg.split.validation_ratio,
        )

        with mlflow.start_run(run_name=f"{market}-catboost-final"):
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("market", market)
            mlflow.set_tag("model_type", cfg.model.name)
            mlflow.set_tag("evaluation_stage", "final_test")

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

            # Train using training data and validation-based early stopping.
            model = train_model(
                train,
                validation,
                iterations=cfg.model.iterations,
                learning_rate=cfg.model.learning_rate,
                depth=cfg.model.depth,
                early_stopping_rounds=cfg.model.early_stopping_rounds,
                random_seed=cfg.model.random_seed,
            )

            # Validation evaluation.
            validation_predictions = model.predict(validation[FEATURE_COLUMNS])

            validation_metrics = calculate_metrics(
                validation["target_price"],
                validation_predictions,
            )

            validation_baseline = calculate_metrics(
                validation["target_price"],
                validation["baseline_price"],
            )

            # Final evaluation on the untouched test set.
            test_predictions = model.predict(test[FEATURE_COLUMNS])

            test_metrics = calculate_metrics(
                test["target_price"],
                test_predictions,
            )

            test_baseline = calculate_metrics(
                test["target_price"],
                test["baseline_price"],
            )

            mlflow.log_metrics(
                {
                    "validation_mae": validation_metrics["mae"],
                    "validation_rmse": validation_metrics["rmse"],
                    "baseline_mae": validation_baseline["mae"],
                    "baseline_rmse": validation_baseline["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_baseline_mae": test_baseline["mae"],
                    "test_baseline_rmse": test_baseline["rmse"],
                    "best_iteration": model.get_best_iteration(),
                }
            )

            # Save the trained model.
            model_path = models_dir / f"{market}_catboost.cbm"
            model.save_model(str(model_path))

            mlflow.log_artifact(
                str(model_path),
                artifact_path="models",
            )

            # Generate and save validation plots.
            plot_paths = create_evaluation_plots(
                model=model,
                validation=validation,
                predictions=validation_predictions,
                feature_columns=FEATURE_COLUMNS,
                market=market,
                output_dir=project_root / "outputs" / "plots",
            )

            for plot_path in plot_paths:
                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="plots",
                )

            test_results.append(
                {
                    "market": market,
                    "train_observations": len(train),
                    "validation_observations": len(validation),
                    "test_observations": len(test),
                    "validation_mae": validation_metrics["mae"],
                    "validation_rmse": validation_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_baseline_mae": test_baseline["mae"],
                    "test_baseline_rmse": test_baseline["rmse"],
                    "best_iteration": model.get_best_iteration(),
                }
            )

            print(f"\nMarket: {market}")
            print(f"Training observations: {len(train):,}")
            print(f"Validation observations: {len(validation):,}")
            print(f"Test observations: {len(test):,}")
            print(f"Validation MAE: {validation_metrics['mae']:.4f} EUR/MWh")
            print(f"Validation RMSE: {validation_metrics['rmse']:.4f} EUR/MWh")
            print(f"Test MAE: {test_metrics['mae']:.4f} EUR/MWh")
            print(f"Test RMSE: {test_metrics['rmse']:.4f} EUR/MWh")
            print(f"Test baseline MAE: {test_baseline['mae']:.4f} EUR/MWh")
            print(f"Test baseline RMSE: {test_baseline['rmse']:.4f} EUR/MWh")
            print(f"Best iteration: {model.get_best_iteration()}")
            print(f"Model saved: {model_path}")
            print("MLflow run logged successfully.")

    # Export final results for both Danish markets.
    results = pd.DataFrame(test_results)
    results_path = results_dir / "final_test_metrics.csv"
    results.to_csv(results_path, index=False)

    print("\n===== FINAL TEST RESULTS =====")
    print(results.to_string(index=False))
    print(f"\nResults saved: {results_path}")
    print("Training and final evaluation completed successfully!")


if __name__ == "__main__":
    main()
