import pandas as pd
from functools import reduce

from src.utils.config import RAW_DIR, INTERMEDIATE_DIR

# How to convert daily series to monthly
# Options: "mean" or "last"
DAILY_TO_MONTHLY = {
    "T10Y2Y": "mean"
}

# Output column names (lowercase)
COL_MAP = {
    "UNRATE": "unrate",
    "CPIAUCSL": "cpi",
    "FEDFUNDS": "fedfunds",
    "INDPRO": "indpro",
    "T10Y2Y": "t10y2y",
}

def load_series(series_id: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{series_id}.csv", parse_dates=["date"])
    df = df.sort_values("date")
    df = df.rename(columns={"value": COL_MAP[series_id]})
    return df[["date", COL_MAP[series_id]]]

def to_monthly(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    # Convert to month start index for a consistent monthly join
    s = df.set_index("date")[col]
    if method == "mean":
        m = s.resample("MS").mean()  # monthly start bins
    elif method == "last":
        m = s.resample("MS").last()
    else:
        raise ValueError("method must be 'mean' or 'last'")
    out = m.reset_index().rename(columns={col: col})
    return out

def main():
    series_ids = list(COL_MAP.keys())
    dfs = []
    for sid in series_ids:
        df = load_series(sid)
        col = COL_MAP[sid]
        # If the series is daily, resample to monthly
        if sid in DAILY_TO_MONTHLY:
            df = to_monthly(df, col, DAILY_TO_MONTHLY[sid])
        else:
            # Ensure monthly series aligns to month-start dates
            df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")
        dfs.append(df)

    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs)
    merged = merged.sort_values("date").reset_index(drop=True)

    outpath = INTERMEDIATE_DIR / "macro_monthly_merged.csv"
    merged.to_csv(outpath, index=False)
    print(f"Saved merged panel -> {outpath} (rows={merged.shape[0]}, cols={merged.shape[1]})")

if __name__ == "__main__":
    main()