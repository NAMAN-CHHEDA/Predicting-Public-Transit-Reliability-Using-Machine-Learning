"""
02_data_cleaning.py
-------------------
Clean the merged VTA GTFS DataFrame produced by 01_data_loading.py.

Steps
-----
1. Check and print missing values for every column
2. Fix GTFS time strings that cross midnight (e.g. 25:30:00)
   → store arrival_sec / departure_sec (total seconds since service-day midnight)
   → also keep human-readable arrival_td / departure_td as pd.Timedelta
3. Join calendar.txt day-of-week service flags (monday … sunday) via service_id
4. Join calendar_dates.txt to flag exception dates (service added / removed)
5. Drop columns that are not useful for ML
6. Print final summary: shape, column list, null counts, dtypes
7. Save to data/cleaned/vta_cleaned.parquet
"""

from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Helper: GTFS time → total seconds (handles hour >= 24)
# ---------------------------------------------------------------------------

def gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert GTFS HH:MM:SS strings to integer seconds since service-day midnight.
    GTFS allows HH >= 24 to represent times after midnight on the next calendar
    day while still belonging to the same service day.
    """
    parts = series.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def gtfs_time_to_timedelta(series: pd.Series) -> pd.Series:
    """Convert GTFS HH:MM:SS strings to pd.Timedelta objects."""
    secs = gtfs_time_to_seconds(series)
    return pd.to_timedelta(secs, unit="s")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded merged DataFrame: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# 2. Missing-value audit
# ---------------------------------------------------------------------------

def audit_missing(df: pd.DataFrame) -> None:
    print(f"\n{SECTION}")
    print("STEP 1 — Missing value audit")
    print(SECTION)

    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    report = pd.DataFrame({"null_count": null_counts, "null_%": null_pct})
    report = report[report["null_count"] > 0].sort_values("null_%", ascending=False)

    if report.empty:
        print("  No missing values found.")
    else:
        print(report.to_string())
    print(f"\n  Total columns with any nulls: {len(report)}")


# ---------------------------------------------------------------------------
# 3. Fix GTFS times
# ---------------------------------------------------------------------------

def fix_gtfs_times(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SECTION}")
    print("STEP 2 — GTFS time conversion (supports hour ≥ 24)")
    print(SECTION)

    for col in ("arrival_time", "departure_time"):
        sec_col = col.replace("_time", "_sec")
        td_col  = col.replace("_time", "_td")

        df[sec_col] = gtfs_time_to_seconds(df[col])
        df[td_col]  = gtfs_time_to_timedelta(df[col])

        over24 = (df[sec_col] >= 86400).sum()
        print(f"  {col:20s} → {sec_col} (int secs)  |  {td_col} (Timedelta)"
              f"  |  rows with hour≥24: {over24:,}")

    # drop the original raw string columns (no longer needed)
    df = df.drop(columns=["arrival_time", "departure_time"])
    print("\n  Dropped original string columns: arrival_time, departure_time")
    return df


# ---------------------------------------------------------------------------
# 4. Join calendar day-of-week flags
# ---------------------------------------------------------------------------

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SECTION}")
    print("STEP 3 — Join calendar.txt day-of-week service flags")
    print(SECTION)

    cal = pd.read_csv(RAW / "calendar.txt", dtype=str)

    # Cast day columns to int (0/1) and date columns to datetime
    for day in DAYS:
        cal[day] = pd.to_numeric(cal[day], errors="coerce").astype("Int8")

    cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d")
    cal["end_date"]   = pd.to_datetime(cal["end_date"],   format="%Y%m%d")

    before = df.shape[1]
    df = df.merge(
        cal[["service_id"] + DAYS + ["start_date", "end_date"]],
        on="service_id",
        how="left",
        validate="many_to_one",
    )
    added = df.shape[1] - before
    unmatched = df["monday"].isnull().sum()

    print(f"  Added {added} columns: {DAYS + ['start_date', 'end_date']}")
    print(f"  service_id rows with no calendar match: {unmatched:,}")

    # calendar_dates-only services have no row in calendar.txt → fill day flags
    # with 0 (service not regularly scheduled on any day of the week)
    for day in DAYS:
        df[day] = df[day].fillna(0).astype("Int8")
    print(f"  Filled {unmatched:,} null day-of-week flag rows with 0")
    return df


# ---------------------------------------------------------------------------
# 5. Flag exception dates from calendar_dates.txt
# ---------------------------------------------------------------------------

def add_calendar_date_flags(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SECTION}")
    print("STEP 4 — Join calendar_dates.txt exception flags")
    print(SECTION)

    cd = pd.read_csv(RAW / "calendar_dates.txt", dtype=str)
    cd["exception_type"] = pd.to_numeric(cd["exception_type"])

    # Count how many times each service_id has service added (type 1) or
    # removed (type 2) — useful as a feature (service stability / holiday flag)
    added_counts   = (
        cd[cd["exception_type"] == 1]
        .groupby("service_id")
        .size()
        .rename("exception_added_count")
    )
    removed_counts = (
        cd[cd["exception_type"] == 2]
        .groupby("service_id")
        .size()
        .rename("exception_removed_count")
    )

    exc = pd.concat([added_counts, removed_counts], axis=1).reset_index()
    exc["exception_added_count"]   = exc["exception_added_count"].fillna(0).astype(int)
    exc["exception_removed_count"] = exc["exception_removed_count"].fillna(0).astype(int)
    exc["has_exception"]           = (
        (exc["exception_added_count"] + exc["exception_removed_count"]) > 0
    ).astype(int)

    before = df.shape[1]
    df = df.merge(exc, on="service_id", how="left", validate="many_to_one")
    df["exception_added_count"]   = df["exception_added_count"].fillna(0).astype(int)
    df["exception_removed_count"] = df["exception_removed_count"].fillna(0).astype(int)
    df["has_exception"]           = df["has_exception"].fillna(0).astype(int)

    added = df.shape[1] - before
    print(f"  Added {added} columns: exception_added_count, exception_removed_count, has_exception")
    print(f"  Services with at least 1 exception: {exc['has_exception'].sum()}")
    print(f"  exception_type=1 (service added)  rows: {(cd['exception_type']==1).sum()}")
    print(f"  exception_type=2 (service removed) rows: {(cd['exception_type']==2).sum()}")
    return df


# ---------------------------------------------------------------------------
# 6. Drop low-utility columns
# ---------------------------------------------------------------------------

# Columns to drop and the reason for dropping each
DROP_COLUMNS = {
    "stop_headsign":       "100 % null in this feed",
    "trip_short_name":     "100 % null in this feed",
    "shape_dist_traveled": "97.9 % null; not needed for ML",
    "route_desc":          "89.8 % null; long-text non-feature",
    "route_url":           "URL string — no predictive signal",
    "route_color":         "hex colour string — cosmetic only",
    "route_text_color":    "hex colour string — cosmetic only",
    "agency_id":           "constant value 'SC' across all rows",
    "block_id":            "operational block — high-cardinality ID, not an ML feature",
    "shape_id":            "shape geometry ID — not an ML feature",
}


def drop_low_utility_columns(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SECTION}")
    print("STEP 5 — Drop low-utility columns")
    print(SECTION)

    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    for col in to_drop:
        print(f"  Dropping  {col:30s}  ← {DROP_COLUMNS[col]}")

    df = df.drop(columns=to_drop)
    print(f"\n  Columns before drop: {len(to_drop) + df.shape[1]}")
    print(f"  Columns after  drop: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 7. Cast remaining columns to appropriate types
# ---------------------------------------------------------------------------

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SECTION}")
    print("STEP 6 — Type casting")
    print(SECTION)

    int_cols = [
        "stop_sequence", "pickup_type", "drop_off_type",
        "timepoint", "direction_id", "bikes_allowed", "wheelchair_accessible",
        "route_type",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")
            print(f"  {col:30s} → Int16")

    category_cols = [
        "route_id", "service_id", "trip_id", "stop_id",
        "route_short_name", "direction_id",
    ]
    for col in category_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype("category")
            print(f"  {col:30s} → category")

    return df


# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{SECTION}")
    print("STEP 7 — Final DataFrame summary")
    print(SECTION)

    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    summary = pd.DataFrame({
        "dtype":      df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "null_%":     (df.isnull().sum() / len(df) * 100).round(2),
        "nunique":    df.nunique(),
    })
    print(summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SECTION)
    print("VTA GTFS — Data Cleaning (02_data_cleaning.py)")
    print(SECTION)

    # 1. Load
    df = load_merged(CLEANED / "stop_times_merged.parquet")

    # 2. Missing value audit
    audit_missing(df)

    # 3. Fix times
    df = fix_gtfs_times(df)

    # 4. Calendar day-of-week flags
    df = add_calendar_flags(df)

    # 5. Exception date flags
    df = add_calendar_date_flags(df)

    # 6. Drop low-utility columns
    df = drop_low_utility_columns(df)

    # 7. Cast types
    df = cast_column_types(df)

    # 8. Final summary
    print_summary(df)

    # 9. Save
    out = CLEANED / "vta_cleaned.parquet"
    df.to_parquet(out, index=False)
    print(f"\n{SECTION}")
    print(f"Saved cleaned DataFrame → {out}")
    print(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    print(SECTION)


if __name__ == "__main__":
    main()
