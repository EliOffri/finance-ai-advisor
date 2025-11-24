from app.utils.ohlcv import load_ohlcv
from app.ml.features import compute_features
from app.utils.db import get_conn
import json

TICKERS = ["AAPL", "MSFT", "SPY"]

def load_features_to_db():
    conn = get_conn()
    with conn:
        cur = conn.cursor()

        for ticker in TICKERS:
            print(f"Processing {ticker}...")

            cur.execute("DELETE FROM technical_features WHERE ticker = %s", (ticker,))

            df = load_ohlcv(ticker)
            df = compute_features(df)

            # Drop rows with NaNs in ANY computed feature
            # Keep only full-valid rows
            df_clean = df.dropna().copy()

            rows = []
            for _, row in df_clean.iterrows():
                feature_cols = [
                    c for c in df_clean.columns
                    if c not in ["ticker", "date", "open", "high", "low", "close", "volume"]
                ]

                feature_dict = {c: row[c] for c in feature_cols}

                rows.append((ticker, row["date"].date(), json.dumps(feature_dict)))

            # Insert into DB
            cur.executemany(
                """
                INSERT INTO technical_features (ticker, date, features)
                VALUES (%s, %s, %s)
                """,
                rows
            )

            print(f"{ticker}: inserted {len(rows)} rows.")

if __name__ == "__main__":
    load_features_to_db()
