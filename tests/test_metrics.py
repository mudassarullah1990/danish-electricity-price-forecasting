import numpy as np
import pytest

from danish_electricity_price_forecasting.commands import (
    calculate_metrics,
)


def test_perfect_predictions():
    """Perfect predictions should produce zero error."""
    actual = np.array([20.0, 30.0, 40.0])
    predicted = np.array([20.0, 30.0, 40.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_known_prediction_errors():
    """Verify MAE and RMSE against manually calculated values."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 34.0])

    metrics = calculate_metrics(actual, predicted)

    # Absolute errors: 2, 2, 4
    # MAE = (2 + 2 + 4) / 3 = 2.6667
    expected_mae = 8 / 3

    # Squared errors: 4, 4, 16
    # RMSE = sqrt((4 + 4 + 16) / 3) = sqrt(8)
    expected_rmse = np.sqrt(8)

    assert metrics["mae"] == pytest.approx(expected_mae)
    assert metrics["rmse"] == pytest.approx(expected_rmse)


def test_negative_electricity_prices():
    """Verify metrics work with negative electricity prices."""
    actual = np.array([-20.0, -10.0, 10.0])
    predicted = np.array([-15.0, -15.0, 15.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["rmse"] == pytest.approx(5.0)
