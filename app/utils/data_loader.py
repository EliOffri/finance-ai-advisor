from pathlib import Path
import pandas as pd

RAW_PATH = Path("app/data/raw")
CLEAN_PATH = Path("app/data/clean")
CLEAN_PATH.mkdir(parents=True, exist_ok=True)

def load_raw(ticker: str) -> pd.DataFrame:
    file_path = RAW_PATH / f"{ticker}.csv"
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure Date is a proper column
    df = df.reset_index()

    # Lowercase all column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Ensure date column exists
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column. Columns: {df.columns}")

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Optional: remove unused columns (dividends/splits)
    drop_cols = {"dividends", "stock_splits"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df




def save_clean(df: pd.DataFrame, ticker: str):
    file_path = CLEAN_PATH / f"{ticker}.clean.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved clean data → {file_path}")
