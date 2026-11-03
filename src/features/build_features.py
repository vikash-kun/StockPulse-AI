from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "RELIANCE_NS_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "RELIANCE_NS_features.csv"


def load_data():
    """Load cleaned historical stock data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def create_features(df):
    """Create technical and statistical features."""

    df = df.copy()

    # 1. Daily percentage return
    df["Daily_Return"] = (
        df["Close"].pct_change(fill_method=None)
    )

    # 2. Previous-day returns
    for lag in [1, 2, 3, 5]:
        df[f"Return_Lag_{lag}"] = (
            df["Daily_Return"].shift(lag)
        )

    # 3. Moving averages
    for window in [5, 10, 20]:
        df[f"MA_{window}"] = (
            df["Close"].rolling(window=window).mean()
        )

    # 4. Price relative to moving averages
    df["Price_MA_5_Ratio"] = (
        df["Close"] / df["MA_5"]
    )

    df["Price_MA_20_Ratio"] = (
        df["Close"] / df["MA_20"]
    )

    # 5. Rolling volatility
    for window in [5, 10, 20]:
        df[f"Volatility_{window}"] = (
            df["Daily_Return"].rolling(window=window).std()
        )

    # 6. Trading volume change
    df["Volume_Change"] = (
        df["Volume"].pct_change(fill_method=None)
    )

    # 7. Volume compared with its 5-day average
    df["Volume_MA_5_Ratio"] = (
        df["Volume"] /
        df["Volume"].rolling(window=5).mean()
    )

    # 8. Daily price range
    df["High_Low_Range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    # 9. Opening-to-closing price movement
    df["Open_Close_Change"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    # 10. Target: next trading day's return
    df["Next_Day_Return"] = (
        df["Close"].shift(-1) / df["Close"] - 1
    )

    # Target: next trading day's direction
    df["Target"] = (
        df["Next_Day_Return"] > 0
    ).astype(int)

    # Remove rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    return df


def save_features(df):
    """Save the engineered dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature dataset saved successfully!")
    print(f"Location: {OUTPUT_FILE}")


def main():

    print("Loading cleaned stock data...")

    df = load_data()

    print(f"Original records: {len(df)}")

    print("\nCreating features...")

    feature_df = create_features(df)

    print("\nFeature engineering completed!")

    print(f"Final records: {len(feature_df)}")
    print(f"Total columns: {len(feature_df.columns)}")

    print("\nAvailable columns:")
    print(feature_df.columns.tolist())

    print("\nTarget distribution:")
    print(feature_df["Target"].value_counts())

    print("\nMissing values:")
    print(feature_df.isnull().sum().sum())

    print("\nFirst five records:")
    print(feature_df.head())

    save_features(feature_df)


if __name__ == "__main__":
    main()

