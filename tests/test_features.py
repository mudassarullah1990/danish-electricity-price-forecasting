import pandas as pd
import pytest

from danish_electricity_price_forecasting.features import create_features


@pytest.fixture
def sample_prices():
    """Create 240 hours of synthetic Danish electricity prices."""
    return pd.DataFrame(
        {
            "utc_timestamp": pd.date_range(
                "2020-01-01", periods=240, freq="h", tz="UTC"
            ),
            "DK_1_price_day_ahead": range(240),
            "DK_2_price_day_ahead": range(1000, 1240),
        }
    )


def test_calendar_features(sample_prices):
    result = create_features(sample_prices, "DK_1")

    first = result.iloc[0]
    timestamp = first["utc_timestamp"]

    assert first["hour"] == timestamp.hour
    assert first["day_of_week"] == timestamp.day_of_week
    assert first["month"] == timestamp.month
    assert first["day_of_year"] == timestamp.day_of_year
    assert first["is_weekend"] == int(timestamp.day_of_week >= 5)


def test_historical_price_lags(sample_prices):
    result = create_features(sample_prices, "DK_1")

    # First valid row corresponds to original index 168.
    first = result.iloc[0]

    assert first["target_price"] == 168
    assert first["price_lag_48h"] == 120
    assert first["price_lag_72h"] == 96
    assert first["price_lag_168h"] == 0
    assert first["baseline_price"] == 0


def test_missing_history_removed(sample_prices):
    result = create_features(sample_prices, "DK_1")

    # 240 observations minus 168 hours of required history.
    assert len(result) == 72
    assert result["price_lag_168h"].notna().all()
    assert result["target_price"].notna().all()


def test_invalid_market(sample_prices):
    with pytest.raises(ValueError, match="Missing price column"):
        create_features(sample_prices, "DK_3")
