from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "RELIANCE_NS_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

# Features used by the machine learning model
FEATURE_COLUMNS = [
    "Daily_Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "MA_5",
    "MA_10",
    "MA_20",
    "Price_MA_5_Ratio",
    "Price_MA_20_Ratio",
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Volume_Change",
    "Volume_MA_5_Ratio",
    "High_Low_Range",
    "Open_Close_Change"
]

TARGET_COLUMN = "Target"


def load_data():
    """Load engineered stock data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nDEBUG: CSV columns:")
    print(df.columns.tolist())

    print("\nDEBUG: First five rows:")
    print(df.head())

    return df


def prepare_datasets(df):
    """Separate features and target, then split chronologically."""

    required_columns = (
        ["Date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only the columns required for modelling
    data = df[required_columns].copy()

    # Check for missing or infinite values
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    numeric_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]]

    if not numeric_data.map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all().all():
        raise ValueError("Dataset contains infinite values.")

    # Separate features and target
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    # Chronological 80:20 split
    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # Preserve dates for later analysis
    train_dates = data["Date"].iloc[:split_index]
    test_dates = data["Date"].iloc[split_index:]

    # Build output datasets
    train_df = X_train.copy()
    train_df.insert(0, "Date", train_dates)
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df.insert(0, "Date", test_dates)
    test_df[TARGET_COLUMN] = y_test

    return train_df, test_df


def save_datasets(train_df, test_df):
    """Save training and testing datasets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nDatasets saved successfully!")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Testing file: {TEST_FILE}")


def main():

    print("Loading engineered stock data...")

    df = load_data()

    print(f"Total records: {len(df)}")

    print("\nPreparing training and testing datasets...")

    train_df, test_df = prepare_datasets(df)

    print("\nData preparation completed!")

    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    print("\nTraining date range:")
    print(train_df["Date"].min(), "to", train_df["Date"].max())

    print("\nTesting date range:")
    print(test_df["Date"].min(), "to", test_df["Date"].max())

    print("\nTraining target distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nTesting target distribution:")
    print(test_df[TARGET_COLUMN].value_counts())

    save_datasets(train_df, test_df)


if __name__ == "__main__":
    main()