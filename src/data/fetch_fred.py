import os
import json
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv

from src.utils.config import RAW_DIR, INTERMEDIATE_DIR

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Start with a defensible, small macro set
SERIES = {
    "UNRATE": "unrate",        # monthly
    "CPIAUCSL": "cpi",         # monthly
    "FEDFUNDS": "fedfunds",    # monthly
    "INDPRO": "indpro",        # monthly
    "T10Y2Y": "t10y2y",         # daily (we'll handle in merge step)
}

def fetch_series(series_id: str, api_key: str, observation_start="1948-01-01") -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    # FRED uses "." for missing
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df

def main():
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FRED_API_KEY in .env")

    metadata = {
        "source": "FRED",
        "pulled_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "series": {},
    }

    for series_id, name in SERIES.items():
        df = fetch_series(series_id, api_key)
        outpath = RAW_DIR / f"{series_id}.csv"
        df.to_csv(outpath, index=False)

        metadata["series"][series_id] = {
            "name": name,
            "rows": int(df.shape[0]),
            "min_date": df["date"].min().strftime("%Y-%m-%d"),
            "max_date": df["date"].max().strftime("%Y-%m-%d"),
        }
        print(f"Saved {series_id} -> {outpath}")

    meta_path = INTERMEDIATE_DIR / "fred_pull_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {meta_path}")

if __name__ == "__main__":
    main()