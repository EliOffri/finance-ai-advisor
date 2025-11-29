import yfinance as yf
from app.utils.db import get_conn
from app.constants.tickers import TICKERS
import pandas as pd


def fetch_from_yf(ticker: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return clean DataFrame."""
    df = yf.Ticker(ticker).history(period="5y", interval="1d")

    if df.empty:
        print(f"[WARN] No data for {ticker}")
        return df

    df = df.reset_index()

    # rename for consistency
    df = df.rename(columns=str.lower)

    # clean data
    df = df.dropna()
    df = df.sort_values("date")

    return df


def insert_to_db(ticker: str, df: pd.DataFrame):
    """Insert OHLCV rows directly into the PostgreSQL ohlcv table."""
    conn = get_conn()
    with conn:
        cur = conn.cursor()

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
            ON CONFLICT (ticker, date) DO NOTHING;
            """,
            rows
        )

    print(f"{ticker}: inserted {len(rows)} new rows.")


def main():
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = fetch_from_yf(ticker)

        if df.empty:
            continue

        insert_to_db(ticker, df)

    print("Done.")


if __name__ == "__main__":
    main()
