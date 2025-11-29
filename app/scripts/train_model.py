import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from app.utils.db import get_conn
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor


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

def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)

        mse = mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        results.append((mse, r2))

        print(f"[Fold {fold}] MSE={mse:.8f}   R²={r2:.5f}")

    avg_mse = sum(m for m, _ in results) / len(results)
    avg_r2  = sum(r for _, r in results) / len(results)

    print("\n=== TimeSeriesSplit Summary ===")
    print(f"Avg MSE: {avg_mse:.8f}")
    print(f"Avg R²:  {avg_r2:.5f}")

    return avg_mse, avg_r2

def run_all_cv_models(X, y):
    print("\n===== Time-Series Cross Validation =====")

    # 1. Linear Regression
    time_series_cv(LinearRegression(), X, y)

    # 2. Random Forest
    time_series_cv(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=42
        ),
        X, y
    )

    # 3. XGBoost
    time_series_cv(
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        X, y
    )

    # 4. Extra Trees (replacing LightGBM)
    time_series_cv(
        ExtraTreesRegressor(
            n_estimators=400,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),
        X, y
    )


def train_and_save_final_models(X_train, y_train, X_test, y_test):
    print("\n===== Training Final Models =====")

    MODEL_DIR = Path("app/models")
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Linear Regression
    lr = build_pipeline(LinearRegression())
    lr.fit(X_train, y_train)
    evaluate_regression(lr, X_test, y_test, "Linear Regression")
    joblib.dump(lr, MODEL_DIR / "linear_regression.pkl")

    # 2. Random Forest
    rf = build_pipeline(RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    ))
    rf.fit(X_train, y_train)
    evaluate_regression(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")

    # 3. XGBoost
    xgb = build_pipeline(XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
    xgb.fit(X_train, y_train)
    evaluate_regression(xgb, X_test, y_test, "XGBoost")
    joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")

    # 4. Extra Trees
    et = build_pipeline(ExtraTreesRegressor(
        n_estimators=400,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ))
    et.fit(X_train, y_train)
    evaluate_regression(et, X_test, y_test, "Extra Trees")
    joblib.dump(et, MODEL_DIR / "extra_trees.pkl")

    print("\nModels saved successfully.")




def main():
    print("Loading technical features...")
    df = load_data()
    print("Loaded rows:", len(df))

    print("Preparing dataset...")
    X, y = prepare_dataset(df)

    # 1. Cross-validation for all models
    run_all_cv_models(X, y)

    # 2. Final train/test split
    X_train, X_test, y_train, y_test = split_time_based(X, y)

    # 3. Train + save final models
    train_and_save_final_models(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()
