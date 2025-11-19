import pandas as pd
from app.utils.db import get_conn

def load_ohlcv(ticker: str) -> pd.DataFrame:
    conn = get_conn()
    query = """
        SELECT date, open, high, low, close, volume
        FROM ohlcv
        WHERE ticker = %s
        ORDER BY date ASC;
    """
    df = pd.read_sql(query, conn, params=[ticker])
    conn.close()
    return df
