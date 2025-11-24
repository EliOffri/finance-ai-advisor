import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. BASIC RETURNS & MOMENTUM
    df["ret_1"] = df["close"].pct_change()
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["roc_10"] = df["close"].pct_change(10)

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (trend + momentum)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 2. TREND INDICATORS
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["price_over_sma10"] = df["close"] / df["sma_10"]
    df["sma10_over_sma50"] = df["sma_10"] / df["sma_50"]

    # 3. VOLATILITY  (Regime detection)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_50"] = df["ret_1"].rolling(50).std()

    # ATR (true range volatility)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # 4. BOLLINGER BANDS (mean reversion)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 5. STOCHASTIC OSCILLATOR (K & D)
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()

    df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # 6. VOLUME FLOW (OBV, MFI)
    # OBV: On-Balance Volume
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # Money Flow Index (MFI)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = tp * df["volume"]

    pos_flow = money_flow.where(tp > tp.shift(1), 0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0)

    df["mfi"] = 100 - (100 / (1 + pos_flow.rolling(14).sum() /
                                   neg_flow.rolling(14).sum()))

    # Volume momentum
    df["volume_change"] = df["volume"].pct_change()
    df["volume_sma20"] = df["volume"].rolling(20).mean()
    df["volume_over_sma20"] = df["volume"] / df["volume_sma20"]

    # FINAL STEP: Drop all NaN rows
    df = df.dropna()

    return df
