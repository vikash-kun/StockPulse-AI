from pathlib import Path

import pandas as pd
import yfinance as yf


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TICKER = "RELIANCE.NS"
PERIOD = "5y"
INTERVAL = "1d"


def download_stock_data(ticker, period, interval):

    print(f"Downloading data for {ticker}...")

    # Download historical stock data
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        multi_level_index=False
    )

    # Check whether data was downloaded
    if df.empty:
        raise ValueError("No stock data received. Check the ticker symbol.")

    # Handle MultiIndex columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert index into a Date column
    df = df.reset_index()

    # Required columns
    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    # Validate columns
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Keep required columns and adjusted closing price if available
    columns_to_keep = required_columns.copy()

    if "Adj Close" in df.columns:
        columns_to_keep.append("Adj Close")

    df = df[columns_to_keep]

    # Clean and sort data
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"])

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate filename
    filename = ticker.replace(".", "_") + ".csv"

    file_path = DATA_DIR / filename

    # Save CSV
    df.to_csv(file_path, index=False)

    print("\nDownload completed successfully!")
    print(f"Stock: {ticker}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Saved at: {file_path}")

    return df


if __name__ == "__main__":

    try:
        stock_data = download_stock_data(
            ticker=TICKER,
            period=PERIOD,
            interval=INTERVAL
        )

        print("\nFirst 5 records:")
        print(stock_data.head())

    except Exception as error:
        print(f"\nError: {error}")