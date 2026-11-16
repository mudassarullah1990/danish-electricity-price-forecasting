from unittest.mock import Mock

import numpy as np
import pandas as pd

from danish_electricity_price_forecasting.evaluate import (
    create_evaluation_plots,
)


def test_create_evaluation_plots(tmp_path):
    """Verify that all three evaluation plots are saved."""

    validation = pd.DataFrame(
        {
            "target_price": [20.0, 25.0, 30.0, 35.0, 40.0],
        }
    )

    predictions = np.array([21.0, 24.0, 29.0, 36.0, 39.0])

    feature_columns = [
        "hour",
        "price_lag_48h",
        "price_lag_168h",
    ]

    model = Mock()
    model.get_feature_importance.return_value = np.array([20.0, 30.0, 50.0])

    output_dir = tmp_path / "evaluation_plots"

    saved_plots = create_evaluation_plots(
        model=model,
        validation=validation,
        predictions=predictions,
        feature_columns=feature_columns,
        market="DK_1",
        output_dir=output_dir,
    )

    expected_files = {
        "DK_1_actual_vs_predicted.png",
        "DK_1_error_distribution.png",
        "DK_1_feature_importance.png",
    }

    assert len(saved_plots) == 3
    assert {path.name for path in saved_plots} == expected_files

    for path in saved_plots:
        assert path.exists()
        assert path.stat().st_size > 0

    model.get_feature_importance.assert_called_once()
