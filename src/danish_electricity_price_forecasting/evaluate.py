"""Evaluation plots for Danish electricity price forecasting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def create_evaluation_plots(
    model,
    validation,
    predictions,
    feature_columns,
    market: str,
    output_dir: Path,
) -> list[Path]:
    """Create and save three validation plots for one market."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = validation["target_price"].to_numpy()
    predicted = np.asarray(predictions)
    residuals = actual - predicted

    saved_plots = []

    # Plot 1: Actual vs predicted electricity prices
    fig, ax = plt.subplots(figsize=(12, 5))
    hours_to_show = min(168, len(actual))

    ax.plot(
        actual[:hours_to_show],
        label="Actual price",
        linewidth=1.5,
    )
    ax.plot(
        predicted[:hours_to_show],
        label="Predicted price",
        linewidth=1.5,
    )
    ax.set_title(f"{market}: Actual vs Predicted Prices")
    ax.set_xlabel("Validation hour")
    ax.set_ylabel("Electricity price (EUR/MWh)")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_path = output_dir / f"{market}_actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    saved_plots.append(plot_path)

    # Plot 2: Prediction error distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(0, color="red", linestyle="--", label="Zero error")
    ax.set_title(f"{market}: Prediction Error Distribution")
    ax.set_xlabel("Actual - predicted (EUR/MWh)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_path = output_dir / f"{market}_error_distribution.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    saved_plots.append(plot_path)

    # Plot 3: CatBoost feature importance
    importances = model.get_feature_importance()
    order = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        np.asarray(feature_columns)[order],
        importances[order],
    )
    ax.set_title(f"{market}: Top Feature Importances")
    ax.set_xlabel("CatBoost feature importance")
    ax.grid(axis="x", alpha=0.3)

    plot_path = output_dir / f"{market}_feature_importance.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    saved_plots.append(plot_path)

    return saved_plots
