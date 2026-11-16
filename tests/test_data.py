import pandas as pd
import pytest

from danish_electricity_price_forecasting.data import load_data


def test_load_data_sorts_timestamps_and_keeps_danish_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"

    sample = pd.DataFrame(
        {
            "utc_timestamp": [
                "2020-01-01T02:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
            ],
            "DK_1_price_day_ahead": [30.0, 10.0, 20.0],
            "DK_2_price_day_ahead": [35.0, 15.0, 25.0],
            "DE_price_day_ahead": [40.0, 50.0, 60.0],
        }
    )

    sample.to_csv(csv_path, index=False)

    result = load_data(csv_path)

    assert len(result) == 3
    assert result["utc_timestamp"].is_monotonic_increasing
    assert str(result["utc_timestamp"].dt.tz) == "UTC"
    assert "DE_price_day_ahead" not in result.columns
    assert result["DK_1_price_day_ahead"].tolist() == [10.0, 20.0, 30.0]


def test_load_data_missing_file(tmp_path):
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(csv_path)


def test_load_data_missing_required_column(tmp_path):
    csv_path = tmp_path / "missing_column.csv"

    sample = pd.DataFrame(
        {
            "utc_timestamp": ["2020-01-01T00:00:00Z"],
            "DK_1_price_day_ahead": [10.0],
        }
    )

    sample.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="DK_2_price_day_ahead"):
        load_data(csv_path)
