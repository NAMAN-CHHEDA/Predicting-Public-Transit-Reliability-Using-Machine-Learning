"""
05_merge_weather.py
-------------------
Join NOAA daily weather observations (San Jose, CA) onto the cleaned VTA
GTFS DataFrame using the service schedule's start_date as the join key.

Weather source : data/raw/4277464.csv  (NOAA GHCN-Daily, station USC00047821)
GTFS source    : data/cleaned/vta_cleaned.parquet
Output         : data/cleaned/vta_with_weather.parquet

New columns added
-----------------
  wx_date        DATE from weather table (validation column)
  prcp           Daily precipitation in inches
  tmax           Daily max temperature (°F)
  tmin           Daily min temperature (°F)
  is_rainy       1 if prcp > 0.1 inches, else 0
  temp_range     tmax - tmin (°F) — a proxy for weather variability

Join key logic
--------------
  GTFS start_date (datetime) is joined to weather DATE (datetime).
  start_date represents the first date of a service window, so every row
  in the same service window gets the weather conditions for that anchor date.
  Rows where start_date is null (calendar_dates-only services) will have
  NaN for all weather columns.
"""

from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
RAW     = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"

WEATHER_FILE = RAW / "4277464.csv"
GTFS_FILE    = CLEANED / "vta_cleaned.parquet"
OUTPUT_FILE  = CLEANED / "vta_with_weather.parquet"

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Step 1 – Load GTFS
# ---------------------------------------------------------------------------

def load_gtfs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"  GTFS shape      : {df.shape[0]:>7,} rows × {df.shape[1]} cols")
    print(f"  start_date range: {df['start_date'].min().date()} → "
          f"{df['start_date'].max().date()}  "
          f"({df['start_date'].nunique()} unique, "
          f"{df['start_date'].isnull().sum()} null)")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Load and prepare weather
# ---------------------------------------------------------------------------

def load_weather(path: Path) -> pd.DataFrame:
    wx = pd.read_csv(path)

    print(f"  Weather shape   : {wx.shape[0]:>7,} rows × {wx.shape[1]} cols")
    print(f"  Station         : {wx['STATION'].iloc[0]}  ({wx['NAME'].iloc[0]})")
    print(f"  DATE range      : {wx['DATE'].min()} → {wx['DATE'].max()}")
    print(f"  PRCP  range     : {wx['PRCP'].min():.2f} – {wx['PRCP'].max():.2f} inches")
    print(f"  TMAX  range     : {wx['TMAX'].min()} – {wx['TMAX'].max()} °F")
    print(f"  TMIN  range     : {wx['TMIN'].min()} – {wx['TMIN'].max()} °F")

    wx["DATE"] = pd.to_datetime(wx["DATE"])

    # Keep only what's needed for the join; drop metadata columns that
    # are constant across the single station and add no predictive value
    wx_slim = (
        wx[["DATE", "PRCP", "TMAX", "TMIN"]]
        .rename(columns={"DATE": "wx_date",
                         "PRCP": "prcp",
                         "TMAX": "tmax",
                         "TMIN": "tmin"})
        .copy()
    )
    return wx_slim


# ---------------------------------------------------------------------------
# Step 3 – Engineer weather features
# ---------------------------------------------------------------------------

def engineer_weather_features(wx: pd.DataFrame) -> pd.DataFrame:
    wx = wx.copy()
    wx["is_rainy"]   = (wx["prcp"] > 0.1).astype("Int8")
    wx["temp_range"] = (wx["tmax"] - wx["tmin"]).astype("Int16")

    rainy_days = wx["is_rainy"].sum()
    print(f"  Rainy days (prcp > 0.1\")  : {rainy_days} / {len(wx)}")
    print(f"  temp_range mean            : {wx['temp_range'].mean():.1f} °F")
    return wx


# ---------------------------------------------------------------------------
# Step 4 – Join
# ---------------------------------------------------------------------------

def merge_weather(df: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    # Normalise join keys to date-only (no time component)
    df = df.copy()
    df["_join_date"] = df["start_date"].dt.normalize()

    merged = df.merge(
        wx,
        left_on="_join_date",
        right_on="wx_date",
        how="left",
        validate="many_to_one",
    ).drop(columns=["_join_date"])

    return merged


# ---------------------------------------------------------------------------
# Step 5 – Join quality report
# ---------------------------------------------------------------------------

def print_join_report(df_orig: pd.DataFrame, merged: pd.DataFrame) -> None:
    wx_cols = ["prcp", "tmax", "tmin", "is_rainy", "temp_range"]

    print(f"\n  Rows before merge : {len(df_orig):>7,}")
    print(f"  Rows after  merge : {len(merged):>7,}  (should be identical)")

    print(f"\n  Weather column null counts after join:")
    for col in ["wx_date"] + wx_cols:
        n = merged[col].isnull().sum()
        pct = n / len(merged) * 100
        print(f"    {col:<14} {n:>6,}  ({pct:.2f}%)")

    # Coverage breakdown
    null_start = df_orig["start_date"].isnull().sum()
    print(f"\n  Note: {null_start:,} rows have null start_date "
          f"(calendar_dates-only services) → weather cols are NaN for those rows")

    # Show weather value distribution for joined rows
    joined = merged.dropna(subset=["prcp"])
    print(f"\n  Joined rows with weather data: {len(joined):,}")
    print(f"  is_rainy distribution:")
    print(merged["is_rainy"].value_counts(dropna=False).rename(
        index={0: "not rainy (0)", 1: "rainy (1)", pd.NA: "null"}
    ).to_string())

    print(f"\n  prcp   stats: mean={merged['prcp'].mean():.3f}\"  "
          f"max={merged['prcp'].max():.3f}\"")
    print(f"  tmax   stats: mean={merged['tmax'].mean():.1f}°F  "
          f"range {merged['tmin'].min()}–{merged['tmax'].max()}°F")
    print(f"  temp_range stats: mean={merged['temp_range'].mean():.1f}°F  "
          f"max={merged['temp_range'].max()}°F")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("05_merge_weather.py — Join NOAA Weather to VTA GTFS")
    print(SECTION)

    # 1. Load GTFS
    print("\n[1] Loading cleaned GTFS …")
    df = load_gtfs(GTFS_FILE)

    # 2. Load weather
    print("\n[2] Loading NOAA weather …")
    wx_raw = load_weather(WEATHER_FILE)

    # 3. Feature engineering on weather table
    print("\n[3] Engineering weather features …")
    wx = engineer_weather_features(wx_raw)

    # 4. Merge
    print("\n[4] Joining weather onto GTFS (left join on start_date = wx_date) …")
    merged = merge_weather(df, wx)
    print(f"  Result shape: {merged.shape[0]:,} rows × {merged.shape[1]} cols")

    # 5. Quality report
    print(f"\n[5] Join quality report")
    print_join_report(df, merged)

    # 6. Sample output
    print(f"\n[6] Sample rows (5 rows, weather columns highlighted):")
    sample_cols = [
        "trip_id", "route_short_name", "stop_id", "stop_sequence",
        "start_date", "wx_date",
        "prcp", "tmax", "tmin", "is_rainy", "temp_range",
    ]
    sample_cols = [c for c in sample_cols if c in merged.columns]
    sample = merged[sample_cols].dropna(subset=["prcp"]).head(5)
    print(sample.to_string(index=False))

    # 7. Final summary
    print(f"\n[7] Final shape: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
    remaining_cols = list(merged.columns)
    print(f"  Columns ({len(remaining_cols)}): {remaining_cols}")

    # 8. Save
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n  Saved → {OUTPUT_FILE}")
    print(SECTION)


if __name__ == "__main__":
    main()
