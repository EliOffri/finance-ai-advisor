import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from app.utils.db import get_conn
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    conn = get_conn()
    query = """
        SELECT t.ticker, t.date, t.features,
               o.open, o.high, o.low, o.close, o.volume
        FROM technical_features t
        JOIN ohlcv o
            ON t.ticker = o.ticker AND t.date = o.date::date
        ORDER BY t.ticker, t.date;
    """

    df = pd.read_sql(query, conn)

    features_df = pd.json_normalize(df["features"])
    df = pd.concat([df, features_df], axis=1)

    df = df.drop(columns=["features"])

    conn.close()
    return df


def build_pipeline(model):
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def prepare_dataset(df: pd.DataFrame):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["target"] = df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    df = df.groupby("ticker", group_keys=False).apply(lambda g: g.iloc[:-1])

    drop_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = df.drop(columns=drop_cols)

    df = df.dropna()

    y = df["target"]
    X = df.drop(columns=["target"])

    return X, y


def evaluate_regression(model, X_test, y_test, name):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n===== {name} =====")
    print(f"R²:          {r2:.4f}")
    print(f"MSE:         {mse:.8f}")
    print(f"Pred mean:   {preds.mean():.6f}")
    print(f"True mean:   {y_test.mean():.6f}")

    long_rate = (preds > 0).mean()
    print(f"Long signal frequency: {long_rate:.2%}")


def split_time_based(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:]
    )


def main():
    print("Loading technical features...")
    df = load_data()
    print("Loaded rows:", len(df))
    print(df.head())

    print("Preparing dataset...")
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_time_based(X, y)

    # LINEAR REGRESSION
    print("Training Linear Regression...")
    from sklearn.linear_model import LinearRegression
    lr_pipeline = build_pipeline(LinearRegression())
    lr_pipeline.fit(X_train, y_train)
    evaluate_regression(lr_pipeline, X_test, y_test, "Linear Regression (Scaled)")

    # RANDOM FOREST REGRESSOR
    print("Training Random Forest Regressor...")
    from sklearn.ensemble import RandomForestRegressor
    rf_pipeline = build_pipeline(RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    ))
    rf_pipeline.fit(X_train, y_train)
    evaluate_regression(rf_pipeline, X_test, y_test, "Random Forest (Scaled)")

    # SAVE PIPELINED MODELS
    print("Saving models...")
    MODEL_DIR = Path("app/models")
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(lr_pipeline, MODEL_DIR / "linear_regression.pkl")
    joblib.dump(rf_pipeline, MODEL_DIR / "random_forest.pkl")

    print("Models saved to app/models/")


if __name__ == "__main__":
    main()
