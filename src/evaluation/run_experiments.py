import pandas as pd
from src.models.registry import get_models
from src.utils.metrics import rmse, mae
from src.utils.config import PROCESSED_DIR

TEST_H = 60  # last 60 months

def main():
    # Choose a lag spec for exogenous models; lag3 is a good default
    df = pd.read_csv(PROCESSED_DIR / "model_input_lag3.csv", parse_dates=["date"])

    y = df["unrate"].astype(float).reset_index(drop=True)
    X = df.drop(columns=["date", "unrate"]).astype(float).reset_index(drop=True)

    y_train, y_test = y[:-TEST_H], y[-TEST_H:]
    X_train, X_test = X[:-TEST_H], X[-TEST_H:]

    results = []

    for model in get_models():
        if "ARIMAX" in model.name or "DirectRidge" in model.name:
            model.fit(y_train, X_train)
            preds = model.predict(TEST_H, X_test)
        else:
            model.fit(y_train)
            preds = model.predict(TEST_H)

        results.append({
            "model": model.name,
            "rmse": rmse(y_test, preds),
            "mae": mae(y_test, preds),
        })

    leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(leaderboard)

    # Save for the “same screen” + reproducible reporting
    outpath = "reports/leaderboard.csv"
    leaderboard.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}")

if __name__ == "__main__":
    main()