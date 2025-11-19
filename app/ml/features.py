import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simple Moving Averages
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # Exponential Moving Averages
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Daily returns (momentum)
    df["daily_return"] = df["close"].pct_change()

    # Volatility (risk)
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()

    return df
