from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "RELIANCE_NS_clean.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed dataset not found. "
            "Complete Day 3 validation first."
        )

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def prepare_data(df):

    # Calculate daily percentage returns
    df["Daily_Return"] = (
        df["Close"].pct_change(fill_method=None) * 100
    )

    # Calculate 20-day rolling volatility
    df["Volatility_20D"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
        * np.sqrt(252)
    )

    return df


def analyze_data(df):

    print("\n" + "=" * 55)
    print("STOCKPULSE AI - EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\n1. BASIC STATISTICS")

    print(df[["Open", "High", "Low", "Close", "Volume"]].describe())

    print("\n2. DAILY RETURN STATISTICS")

    print(df["Daily_Return"].describe())

    print("\n3. HIGHEST CLOSING PRICE")

    highest = df.loc[df["Close"].idxmax()]

    print("Date:", highest["Date"])
    print("Price:", highest["Close"])

    print("\n4. LOWEST CLOSING PRICE")

    lowest = df.loc[df["Close"].idxmin()]

    print("Date:", lowest["Date"])
    print("Price:", lowest["Close"])

    print("\n5. AVERAGE TRADING VOLUME")

    print(df["Volume"].mean())

    print("\n6. CORRELATION MATRIX")

    print(
        df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].corr().round(3)
    )


def plot_closing_price(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Close"],
        label="Closing Price"
    )

    plt.title("Reliance Industries - Closing Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (INR)")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "closing_price.png")

    plt.close()


def plot_daily_returns(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Daily_Return"],
        label="Daily Returns"
    )

    plt.axhline(y=0, linestyle="--")

    plt.title("Reliance Industries - Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "daily_returns.png")

    plt.close()


def plot_volume(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Volume"],
        label="Trading Volume"
    )

    plt.title("Reliance Industries - Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "trading_volume.png")

    plt.close()


def plot_price_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["Close"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Closing Price Distribution")
    plt.xlabel("Closing Price (INR)")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "price_distribution.png")

    plt.close()


def plot_volatility(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Date"],
        df["Volatility_20D"],
        label="20-Day Annualized Volatility"
    )

    plt.title("Reliance Industries - Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility (%)")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "volatility.png")

    plt.close()


def plot_correlation(df):

    columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    correlation = df[columns].corr()

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(columns)),
        columns
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    for i in range(len(columns)):
        for j in range(len(columns)):

            plt.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Stock Market Feature Correlation")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "correlation.png")

    plt.close()


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading processed stock data...")

    df = load_data()

    print("Preparing analytical features...")

    df = prepare_data(df)

    analyze_data(df)

    print("\nGenerating visualizations...")

    plot_closing_price(df)
    plot_daily_returns(df)
    plot_volume(df)
    plot_price_distribution(df)
    plot_volatility(df)
    plot_correlation(df)

    print("\nEDA COMPLETED SUCCESSFULLY!")

    print("Visualizations saved at:", OUTPUT_DIR)


if __name__ == "__main__":

    main()