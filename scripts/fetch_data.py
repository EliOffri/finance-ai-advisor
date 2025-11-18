import yfinance as yf
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("app/data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "SPY"]

def fetch_ticker(ticker: str):
    t = yf.Ticker(ticker)
    data = t.history(period="5y", interval="1d")

    # Ensure a clean index
    data = data.reset_index()

    file_path = RAW_DATA_PATH / f"{ticker}.csv"
    data.to_csv(file_path, index=False)
    print(f"Saved {ticker} to {file_path}")


def main():
    for ticker in TICKERS:
        fetch_ticker(ticker)

if __name__ == "__main__":
    main()
