import pandas as pd

from src.utils.config import INTERMEDIATE_DIR, PROCESSED_DIR

TARGET = "unrate"

def build_lagged_data(lags=(1, 3, 6)):
    """
    Create lagged predictor datasets for forecasting UNRATE. 

    Parameters
    ----------
    lags : tuple of int
        Number of months to lag predictors (e.g. 1, 3, 6).
    """
    df = pd.read_csv(
        INTERMEDIATE_DIR / "macro_monthly_merged.csv",
        parse_dates=["date"],
    ).sort_values("date")

    predictors = [c for c in df.columns if c not in ["date", TARGET]]

    for L in lags: 
        out = df[["date", TARGET]].copy()

        for p in predictors:
            out[f"{p}_lag{L}"] = df[p].shift(L)

        out = out.dropna().reset_index(drop=True)

        outpath = PROCESSED_DIR / f"model_input_lag{L}.csv"
        out.to_csv(outpath, index=False)

        print(f"[build_lags] Saved {outpath} with shape {out.shape}")

if __name__ == "__main__":
    build_lagged_data()