from app.utils.data_loader import load_raw, clean_data, save_clean
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "SPY"]

def main():
    for ticker in TICKERS:
        print(f"Cleaning {ticker}...")
        df = load_raw(ticker)
        df_clean = clean_data(df)
        save_clean(df_clean, ticker)

if __name__ == "__main__":
    main()
