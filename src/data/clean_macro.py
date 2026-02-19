import pandas as pd
from pathlib import Path

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Remove any non-numeric characters (except . and -), then convert to float.
    If the column is already numeric, returns it unchanged.
    """
    # If already numeric, don't stringify + reconvert (keeps it stable)
    if pd.api.types.is_numeric_dtype(series):
        return series

    s = series.astype(str)

    # Keep only digits, decimal point, minus sign
    s = s.str.replace(r"[^0-9\.-]", "", regex=True)

    return pd.to_numeric(s, errors="coerce")


def main():
    input_path = Path("data/intermediate/macro_monthly_merged.csv")
    output_path = Path("data/processed/macro_monthly_clean.csv")

    df = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date").set_index("date")

    print("\nDtypes BEFORE cleaning:\n", df.dtypes)
    print("\nMissing values BEFORE cleaning:\n", df.isna().sum())

    # Show a quick sample BEFORE cleaning (first 3 rows)
    print("\nSample BEFORE cleaning:\n", df.head(3))

    # Clean columns
    for col in df.columns:
        df[col] = clean_numeric_column(df[col])

    print("\nDtypes AFTER numeric coercion:\n", df.dtypes)

    # Time interpolation + drop edges
    df = df.interpolate(method="time").dropna()

    # Round unrate (case-insensitive)
    unrate_cols = [c for c in df.columns if c.strip().lower() == "unrate"]
    if unrate_cols:
        unrate_col = unrate_cols[0]
        df[unrate_col] = df[unrate_col].round(1)
        print(f"\nRounded {unrate_col} to 1 decimal.")
    else:
        print("\nWARNING: Could not find an UNRATE/unrate column to round.")

    print("\nMissing values AFTER cleaning:\n", df.isna().sum())
    print("\nSample AFTER cleaning:\n", df.head(3))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nSaved cleaned data to: {output_path.resolve()}")

if __name__ == "__main__":
    main()