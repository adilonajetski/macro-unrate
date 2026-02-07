import pandas as pd
from pathlib import Path

def main():
    input_path = Path("data/intermediate/macro_monthly_merged.csv")
    output_path = Path("data/processed/macro_monthly_clean.csv")

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    # Inspect missingness (prints once, then you forget about it)
    print("Missing values before cleaning:")
    print(df.isna().sum())

    # Time-aware interpolation
    df = df.interpolate(method="time")

    # Drop any remaining edge NaNs
    df = df.dropna()

    print("Missing values after cleaning:")
    print(df.isna().sum())

    df.to_csv(output_path)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()