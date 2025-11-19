from app.utils.ohlcv import load_ohlcv
from app.ml.features import compute_features
from app.utils.db import get_conn
import pandas as pd

TICKERS = ["AAPL", "MSFT", "SPY"]

def load_features_to_db():
    conn = get_conn()
    with conn:
        cur = conn.cursor()

        for ticker in TICKERS:
            print(f"Processing {ticker}...")

            df = load_ohlcv(ticker)
            df = compute_features(df)

            # Only insert rows with full feature data (after window periods)
            rows = [
                (
                    ticker,
                    row["date"],
                    row["sma_10"], row["sma_20"], row["sma_50"],
                    row["ema_10"], row["ema_20"], row["ema_50"],
                    row["daily_return"], row["volatility_20"],
                )
                for _, row in df.iterrows()
                if pd.notnull(row["sma_50"])
            ]

            cur.executemany(
                """
                INSERT INTO technical_features (
                    ticker, date,
                    sma_10, sma_20, sma_50,
                    ema_10, ema_20, ema_50,
                    daily_return, volatility_20
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows
            )

            print(f"{ticker}: inserted {len(rows)} rows")

if __name__ == "__main__":
    load_features_to_db()
