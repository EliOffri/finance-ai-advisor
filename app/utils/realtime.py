import pandas as pd
from app.utils.db import get_conn
from app.ml.features import compute_features

# The exact order your model expects:
FEATURE_ORDER = [
    'sma_10',
    'sma_20',
    'sma_50',
    'ema_10',
    'ema_20',
    'ema_50',
    'daily_return',
    'volatility_20'
]


def get_latest_ohlcv(ticker: str):
    """
    Fetch enough history (60 rows) to compute SMA_50, EMA_50, volatility_20, etc.
    """
    conn = get_conn()

    query = """
        SELECT *
        FROM ohlcv
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT 60;
    """

    df = pd.read_sql(query, conn, params=[ticker])
    conn.close()

    if df.empty:
        return None

    # Ensure chronological order (important!)
    df = df.sort_values("date")
    return df


def compute_latest_features(ticker: str):
    """
    Fetch OHLCV, compute features, and return a model-ready vector
    for the latest available day.
    """
    df = get_latest_ohlcv(ticker)
    if df is None or len(df) < 50:
        return None  # not enough data for SMA_50 / EMA_50

    df_feat = compute_features(df)

    # Most recent feature row
    last_row = df_feat.iloc[-1]

    # Build feature vector in the correct order
    feature_vector = [float(last_row[col]) for col in FEATURE_ORDER]

    return feature_vector
