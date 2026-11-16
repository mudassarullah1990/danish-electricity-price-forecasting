import pandas as pd
import pytest

from danish_electricity_price_forecasting.train import split_data


@pytest.fixture
def sample_data():
    """Create 20 complete days of synthetic hourly data."""
    return pd.DataFrame(
        {
            "utc_timestamp": pd.date_range(
                "2020-01-01",
                periods=20 * 24,
                freq="h",
                tz="UTC",
            ),
            "target_price": range(20 * 24),
        }
    )


def test_chronological_split(sample_data):
    train, validation, test = split_data(sample_data)

    assert len(train) == 14 * 24
    assert len(validation) == 3 * 24
    assert len(test) == 3 * 24

    assert train["utc_timestamp"].max() < validation["utc_timestamp"].min()
    assert validation["utc_timestamp"].max() < test["utc_timestamp"].min()


def test_complete_days_only(sample_data):
    train, validation, test = split_data(sample_data)

    for subset in (train, validation, test):
        daily_counts = subset.groupby(subset["utc_timestamp"].dt.floor("D")).size()

        assert (daily_counts == 24).all()


def test_incomplete_day_removed(sample_data):
    # Remove one hour from the first day.
    incomplete = sample_data.drop(index=0)

    train, validation, test = split_data(incomplete)

    combined = pd.concat([train, validation, test])

    assert len(combined) == 19 * 24
    assert pd.Timestamp("2020-01-01", tz="UTC") not in set(
        combined["utc_timestamp"].dt.floor("D")
    )


@pytest.mark.parametrize(
    "train_ratio,validation_ratio",
    [
        (0, 0.15),
        (-0.1, 0.15),
        (0.70, 0),
        (0.90, 0.15),
    ],
)
def test_invalid_split_ratios(sample_data, train_ratio, validation_ratio):
    with pytest.raises(ValueError):
        split_data(
            sample_data,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
