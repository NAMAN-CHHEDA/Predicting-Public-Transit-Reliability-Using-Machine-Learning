"""
04c_calculate_delays.py
-----------------------
Estimate actual bus delays by comparing the GTFS-RT vehicle position timestamp
(when the GPS unit last reported near a stop) against the GTFS static scheduled
arrival time for that same stop.

Methodology
-----------
  1. Loop over ALL data/raw/vehicle_positions_*.csv snapshots.
  2. For each snapshot, join on (trip_id, current_stop_sequence) to GTFS static.
  3. Convert vehicle_timestamp (POSIX UTC) to seconds-since-service-day-midnight
     using the PDT offset (UTC−7, active in April 2026).
  4. actual_delay_seconds = vehicle_timestamp_sod − scheduled_arrival_sec
     • Positive = vehicle reported near the stop AFTER scheduled arrival → late
     • Negative = vehicle reported BEFORE scheduled arrival → early / not yet arrived
  5. delay_minutes = actual_delay_seconds / 60
  6. is_delayed = 1 if delay_minutes > 5
  7. Stack all per-snapshot results, then deduplicate on (trip_id, stop_sequence)
     keeping the observation with the LARGEST absolute delay (worst-case per stop).

Caveat
------
  vehicle_timestamp is the time of the last GPS position report, NOT a confirmed
  stop-arrival time.  Treat delay values as an approximation; they are most
  reliable for STOPPED_AT status rows.

Input
-----
  data/raw/vehicle_positions_*.csv   (all snapshot files)
  data/cleaned/vta_cleaned.parquet

Output
------
  data/raw/calculated_delays_all.csv
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW     = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"

GTFS_FILE = CLEANED / "vta_cleaned.parquet"
OUT_FILE  = RAW / "calculated_delays_all.csv"

# PDT = UTC−7 (active April 2026 in the San Jose service area)
PDT_OFFSET_SEC = 7 * 3600

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Step 1 – Load
# ---------------------------------------------------------------------------

def load_vehicle_positions(path: Path) -> pd.DataFrame:
    vp = pd.read_csv(path)
    before = len(vp)
    vp = vp.dropna(subset=["trip_id", "current_stop_sequence"])
    dropped = before - len(vp)
    vp["trip_id"]               = vp["trip_id"].astype(int).astype(str)
    vp["current_stop_sequence"] = vp["current_stop_sequence"].astype(int)
    print(f"    {path.name}: {before} rows total, "
          f"{dropped} dropped (no trip), {len(vp)} for join")
    return vp


def load_gtfs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trip_id"]       = df["trip_id"].astype(str)
    df["stop_sequence"] = df["stop_sequence"].astype(int)
    print(f"  GTFS rows: {len(df):,}  |  "
          f"unique trips: {df['trip_id'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Convert vehicle_timestamp → seconds since service-day midnight
# ---------------------------------------------------------------------------

def vehicle_ts_to_sod(posix_ts: pd.Series) -> pd.Series:
    """
    Convert a POSIX UTC timestamp to seconds-since-midnight in PDT (UTC−7).
    Uses modulo 86400 so it wraps correctly even for late-night service.
    """
    local_epoch = posix_ts - PDT_OFFSET_SEC   # shift to local clock
    return local_epoch % 86400                 # seconds since local midnight


# ---------------------------------------------------------------------------
# Step 3 – Join
# ---------------------------------------------------------------------------

def join_vp_to_gtfs(vp: pd.DataFrame, gtfs: pd.DataFrame) -> pd.DataFrame:
    # GTFS columns we need for the delay calc and the report
    gtfs_cols = [
        "trip_id", "stop_sequence", "stop_id",
        "route_id", "route_short_name", "route_long_name",
        "arrival_sec", "departure_sec",
        "direction_id", "trip_headsign",
    ]

    merged = vp.merge(
        gtfs[gtfs_cols].drop_duplicates(subset=["trip_id", "stop_sequence"]),
        left_on=["trip_id", "current_stop_sequence"],
        right_on=["trip_id", "stop_sequence"],
        how="inner",
    )
    return merged


# ---------------------------------------------------------------------------
# Step 4 – Compute delay features
# ---------------------------------------------------------------------------

def compute_delays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Seconds since midnight for the vehicle GPS report
    df["vehicle_ts_sod"] = vehicle_ts_to_sod(df["vehicle_timestamp"])

    # Delay = actual observed time at stop − scheduled arrival time
    df["actual_delay_seconds"] = (
        df["vehicle_ts_sod"] - df["arrival_sec"]
    ).round(1)

    df["delay_minutes"] = (df["actual_delay_seconds"] / 60).round(2)

    # Flag: late by more than 5 minutes
    df["is_delayed"] = (df["delay_minutes"] > 5).astype("Int8")

    return df


# ---------------------------------------------------------------------------
# Step 5 – Report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    total       = len(df)
    n_delayed   = df["is_delayed"].eq(1).sum()
    n_early     = df["delay_minutes"].lt(-5).sum()
    n_on_time   = total - n_delayed - n_early
    pct_delayed = n_delayed / total * 100
    avg_delay   = df["delay_minutes"].mean()
    median_delay = df["delay_minutes"].median()

    print(f"\n{SECTION}")
    print("Delay Calculation Report")
    print(SECTION)
    print(f"  Total matched rows            : {total:>6}")
    print(f"  Late  (> +5 min)  is_delayed=1: {n_delayed:>6}  ({pct_delayed:.1f}%)")
    print(f"  Early (< −5 min)              : {n_early:>6}  ({n_early/total*100:.1f}%)")
    print(f"  On-time (−5 to +5 min)        : {n_on_time:>6}  ({n_on_time/total*100:.1f}%)")
    print(f"\n  Average delay   : {avg_delay:>7.2f} min")
    print(f"  Median  delay   : {median_delay:>7.2f} min")
    print(f"  Max     delay   : {df['delay_minutes'].max():>7.2f} min")
    print(f"  Min     delay   : {df['delay_minutes'].min():>7.2f} min")

    # Status breakdown
    print(f"\n  By current_status_name:")
    for status, grp in df.groupby("current_status_name"):
        gmean = grp["delay_minutes"].mean()
        print(f"    {status:<22} {len(grp):>3} rows  avg delay: {gmean:+.2f} min")

    # Top 5 most delayed routes (min 3 matched vehicles)
    print(f"\n  Top 5 most delayed routes (mean delay_minutes, ≥3 vehicles):")
    route_stats = (
        df.groupby("route_short_name", observed=True)
        .agg(
            mean_delay  =("delay_minutes",  "mean"),
            median_delay=("delay_minutes",  "median"),
            n_vehicles  =("vehicle_id",     "count"),
            pct_delayed =("is_delayed",     "mean"),
        )
        .query("n_vehicles >= 3")
        .sort_values("mean_delay", ascending=False)
        .head(5)
    )
    if route_stats.empty:
        print("    (No routes with ≥3 matched vehicles)")
    else:
        print(f"    {'Route':<14} {'Mean (min)':>11} {'Median':>8} "
              f"{'Vehicles':>9} {'% Late':>7}")
        print(f"    {'-'*52}")
        for route, row in route_stats.iterrows():
            print(f"    {str(route):<14} {row['mean_delay']:>+11.2f} "
                  f"{row['median_delay']:>+8.2f} "
                  f"{int(row['n_vehicles']):>9} "
                  f"{row['pct_delayed']*100:>6.1f}%")

    # Top 5 earliest routes
    print(f"\n  Top 5 earliest routes (most negative mean delay, ≥3 vehicles):")
    route_early = (
        df.groupby("route_short_name", observed=True)
        .agg(
            mean_delay=("delay_minutes", "mean"),
            n_vehicles=("vehicle_id",    "count"),
        )
        .query("n_vehicles >= 3")
        .sort_values("mean_delay", ascending=True)
        .head(5)
    )
    if route_early.empty:
        print("    (No routes with ≥3 matched vehicles)")
    else:
        for route, row in route_early.iterrows():
            print(f"    {str(route):<14}  {row['mean_delay']:>+8.2f} min  "
                  f"({int(row['n_vehicles'])} vehicles)")

    print(f"\n  Note: delay = vehicle_timestamp (GPS report time) − scheduled arrival.")
    print(f"        Most reliable for STOPPED_AT rows; treat as approximation.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("04c_calculate_delays.py — Delays Across All Vehicle Position Snapshots")
    print(SECTION)

    # Discover all snapshot files
    vp_files = sorted(RAW.glob("vehicle_positions_*.csv"))
    if not vp_files:
        print("  ERROR: No vehicle_positions_*.csv files found in data/raw/")
        return
    print(f"\n  Found {len(vp_files)} snapshot file(s):")

    # Load GTFS once
    print("\n[1] Loading GTFS static …")
    gtfs = load_gtfs(GTFS_FILE)

    # -----------------------------------------------------------------------
    # Loop: process each snapshot file individually
    # -----------------------------------------------------------------------
    all_frames: list[pd.DataFrame] = []
    per_file_stats: list[dict] = []

    print(f"\n[2] Processing each snapshot …")
    for vp_file in vp_files:
        vp = load_vehicle_positions(vp_file)
        if vp.empty:
            print(f"    {vp_file.name}: no usable rows — skipping")
            continue

        merged = join_vp_to_gtfs(vp, gtfs)
        if merged.empty:
            print(f"    {vp_file.name}: 0 GTFS matches — skipping")
            continue

        result = compute_delays(merged)

        snap_utc = datetime.fromtimestamp(
            result["vehicle_timestamp"].iloc[0], tz=timezone.utc
        )
        snap_pdt = snap_utc - timedelta(hours=7)

        per_file_stats.append({
            "file":        vp_file.name,
            "snapshot_pdt": snap_pdt.strftime("%Y-%m-%d %H:%M:%S"),
            "matched":     len(result),
            "delayed":     int(result["is_delayed"].eq(1).sum()),
            "avg_delay":   round(result["delay_minutes"].mean(), 2),
            "max_delay":   round(result["delay_minutes"].max(), 2),
        })

        all_frames.append(result)

    if not all_frames:
        print("\n  No data could be processed. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Stack all snapshots
    # -----------------------------------------------------------------------
    print(f"\n[3] Stacking {len(all_frames)} snapshots …")
    stacked = pd.concat(all_frames, ignore_index=True)
    print(f"  Stacked total rows (before dedup): {len(stacked):,}")

    # -----------------------------------------------------------------------
    # Dedup: per (trip_id, stop_sequence) keep row with largest |delay|
    # -----------------------------------------------------------------------
    print(f"\n[4] Deduplicating on (trip_id, stop_sequence) "
          f"→ keep row with largest absolute delay …")
    stacked["_abs_delay"] = stacked["delay_minutes"].abs()
    stacked = (
        stacked
        .sort_values("_abs_delay", ascending=False)
        .drop_duplicates(subset=["trip_id", "stop_sequence"], keep="first")
        .drop(columns=["_abs_delay"])
        .reset_index(drop=True)
    )
    print(f"  After dedup: {len(stacked):,} unique (trip_id, stop_sequence) pairs")

    # -----------------------------------------------------------------------
    # Per-file summary table
    # -----------------------------------------------------------------------
    print(f"\n[5] Per-snapshot summary:")
    stats_df = pd.DataFrame(per_file_stats)
    print(f"  {'File':<30} {'Time (PDT)':<22} {'Matched':>8} "
          f"{'Delayed':>8} {'% Late':>7} {'Avg(min)':>9} {'Max(min)':>9}")
    print(f"  {'-'*85}")
    for _, row in stats_df.iterrows():
        pct = row["delayed"] / row["matched"] * 100 if row["matched"] > 0 else 0
        print(f"  {row['file']:<30} {row['snapshot_pdt']:<22} "
              f"{int(row['matched']):>8,} {int(row['delayed']):>8,} "
              f"{pct:>6.1f}% {row['avg_delay']:>+9.2f} {row['max_delay']:>+9.2f}")

    # -----------------------------------------------------------------------
    # Combined report
    # -----------------------------------------------------------------------
    print(f"\n[6] Combined report across all snapshots (after dedup):")
    print_report(stacked)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_cols = [
        "vehicle_id", "trip_id", "route_id", "route_short_name",
        "stop_sequence", "stop_id", "current_status_name",
        "latitude", "longitude", "speed_mph", "occupancy_label",
        "vehicle_timestamp", "vehicle_ts_sod",
        "arrival_sec", "departure_sec",
        "actual_delay_seconds", "delay_minutes", "is_delayed",
        "snapshot_ts",
    ]
    out_cols = [c for c in out_cols if c in stacked.columns]
    stacked[out_cols].to_csv(OUT_FILE, index=False)
    print(f"\n[7] Saved {len(stacked):,} rows × {len(out_cols)} cols → {OUT_FILE.name}")
    print(SECTION)

    # Preview top delayed
    print("\nTop 10 most delayed observations (all snapshots combined):")
    preview_cols = ["vehicle_id", "route_short_name", "stop_sequence",
                    "current_status_name", "delay_minutes", "is_delayed",
                    "speed_mph", "snapshot_ts"]
    preview_cols = [c for c in preview_cols if c in stacked.columns]
    print(stacked[preview_cols]
          .sort_values("delay_minutes", ascending=False)
          .head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
