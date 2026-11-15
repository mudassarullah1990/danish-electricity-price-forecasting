from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = (
    "https://data.open-power-system-data.org/time_series/"
    "2020-10-06/time_series_60min_singleindex.csv"
)


def download_data(destination: Path) -> Path:
    """Download the OPSD dataset if it is not already available."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.is_file():
        print("Downloading OPSD electricity dataset...")
        urlretrieve(DATA_URL, destination)
        print(f"Dataset downloaded to {destination}")

    return destination


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load hourly Danish electricity data from the OPSD CSV."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        usecols=lambda column: column == "utc_timestamp" or column.startswith("DK_"),
    )

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)

    df = df.sort_values("utc_timestamp").reset_index(drop=True)

    required_columns = [
        "DK_1_price_day_ahead",
        "DK_2_price_day_ahead",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df
