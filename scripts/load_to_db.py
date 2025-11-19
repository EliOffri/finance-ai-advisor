import pandas as pd
from app.utils.db import get_conn
from pathlib import Path

CLEAN_PATH = Path("app/data/clean")
TICKERS = ["AAPL", "MSFT", "SPY"]

def load_csv(ticker: str):
    file_path = CLEAN_PATH / f"{ticker}.clean.csv"
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

def load_to_db():
    conn = get_conn()
    with conn:
        cur = conn.cursor()

        for ticker in TICKERS:
            print(f"Loading {ticker}...")

            df = load_csv(ticker)

            rows = [
                (
                    ticker,
                    row["date"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                )
                for _, row in df.iterrows()
            ]

            cur.executemany(
                """
                INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                rows
            )

            print(f"{ticker} loaded! ({len(rows)} rows)")

if __name__ == "__main__":
    load_to_db()
