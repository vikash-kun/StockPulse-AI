from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "RELIANCE_NS.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_FILE = PROCESSED_DIR / "RELIANCE_NS_clean.csv"


def validate_stock_data():

    print("=" * 50)
    print("STOCKPULSE AI - DATA VALIDATION")
    print("=" * 50)

    # 1. Check whether the dataset exists
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Complete Day 2 data downloading first."
        )

    # 2. Load dataset
    df = pd.read_csv(RAW_FILE)

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("\n1. DATASET INFORMATION")

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # 3. Check required columns
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Convert columns into correct data types
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if "Adj Close" in df.columns:
        numeric_columns.append("Adj Close")

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    print("\n2. DATA TYPES")

    print(df.dtypes)

    # 5. Check missing values
    print("\n3. MISSING VALUES")

    print(df.isnull().sum())

    # 6. Check duplicate records
    print("\n4. DUPLICATE RECORDS")

    print(
        "Exact duplicate rows:",
        df.duplicated().sum()
    )

    print(
        "Duplicate dates:",
        df["Date"].duplicated().sum()
    )

    # 7. Check invalid numerical values
    print("\n5. INVALID VALUES")

    invalid_prices = (
        df[["Open", "High", "Low", "Close"]] <= 0
    ).any(axis=1)

    negative_volume = df["Volume"] < 0

    print("Invalid price rows:", invalid_prices.sum())

    print("Negative volume rows:", negative_volume.sum())

    # 8. Validate OHLC relationships
    print("\n6. OHLC VALIDATION")

    invalid_ohlc = (
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    )

    print("Invalid OHLC rows:", invalid_ohlc.sum())

    # 9. Check date ordering
    print("\n7. DATE VALIDATION")

    print(
        "Dates already sorted:",
        df["Date"].is_monotonic_increasing
    )

    print("First date:", df["Date"].min())

    print("Last date:", df["Date"].max())

    # 10. Check whether the dataset passes validation
    has_errors = (
        df[required_columns].isnull().any().any()
        or df["Date"].duplicated().any()
        or invalid_prices.any()
        or negative_volume.any()
        or invalid_ohlc.any()
    )

    if has_errors:

        print("\nVALIDATION FAILED!")

        print(
            "Inspect the reported problems before "
            "using this dataset."
        )

        return

    # 11. Sort data chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # 12. Save validated data
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_FILE, index=False)

    print("\n" + "=" * 50)
    print("VALIDATION SUCCESSFUL!")
    print("=" * 50)

    print("Total records:", len(df))

    print("Validated file:", PROCESSED_FILE)


if __name__ == "__main__":

    try:
        validate_stock_data()

    except Exception as error:
        print("\nERROR:", error)