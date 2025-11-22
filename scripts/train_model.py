# app/scripts/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from app.utils.db import get_conn


def load_data():
    """
    Load feature-engineered stock data from PostgreSQL.
    This mirrors how load_to_db() works by using get_conn().
    """
    conn = get_conn()
    query = """
        SELECT t.*, o.open, o.high, o.low, o.close, o.volume
        FROM technical_features t
        JOIN ohlcv o ON t.ticker = o.ticker AND t.date = o.date
        ORDER BY t.ticker, t.date;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def prepare_dataset(df: pd.DataFrame):
    """
    Create the target label and clean dataset for machine learning.
    """

    # Target: 1 if tomorrow's close > today's close
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop last row per ticker (because shift(-1) gives NaN)
    df = df.groupby("ticker").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Columns we should NOT feed into ML
    drop_cols = [
        "id",
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df.drop(columns=drop_cols)

    # Remove remaining NaNs
    df = df.dropna()

    # Feature matrix and target vector
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


def split_time_based(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\n===== {model_name} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)


def main():
    print("Loading technical features from DB...")
    df = load_data()

    print("Preparing dataset...")
    X, y = prepare_dataset(df)

    X_train, X_test, y_train, y_test = split_time_based(X, y)

    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")


if __name__ == "__main__":
    main()
