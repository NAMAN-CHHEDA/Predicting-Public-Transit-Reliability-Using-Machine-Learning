"""
06_merge_realtime.py
--------------------
Merge all collected GTFS-RT TripUpdate snapshots with the cleaned+weather
GTFS static DataFrame, compute delay features, and produce the final
modelling dataset.

Inputs
------
  data/raw/realtime_snapshot_*.csv   — one or more snapshots from 04_fetch_realtime.py
  data/cleaned/vta_with_weather.parquet

Join key
--------
  (trip_id, stop_id)  — stop-level join; avoids row explosion and gives
  the most granular delay signal.  GTFS rows whose (trip_id, stop_id) pair
  was not active in any snapshot get NaN for all RT columns.

New columns
-----------
  rt_snapshot_ts    timestamp of the realtime snapshot (UTC ISO-8601)
  rt_arrival_time   absolute arrival time from RT feed (POSIX seconds)
  rt_arrival_delay  arrival delay in seconds  (raw from feed)
  rt_departure_delay departure delay in seconds (raw from feed)
  rt_vehicle_id     vehicle label from feed
  delay_minutes     rt_arrival_delay / 60  (float, NaN if no RT match)
  is_delayed        1 if delay_minutes > 5 else 0  (0 for NaN rows)

Outputs
-------
  data/cleaned/vta_final.parquet
  data/cleaned/vta_final.csv
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

GTFS_FILE   = CLEANED / "vta_with_weather.parquet"
OUT_PARQUET = CLEANED / "vta_final.parquet"
OUT_CSV     = CLEANED / "vta_final.csv"

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Step 1 – Load & combine all realtime snapshots
# ---------------------------------------------------------------------------

def load_realtime_snapshots(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("realtime_snapshot_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No realtime_snapshot_*.csv files found in {raw_dir}. "
            "Run 04_fetch_realtime.py first."
        )

    print(f"  Found {len(files)} snapshot file(s):")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        print(f"    {f.name:35s}  {len(df):>5,} rows  "
              f"snapshot_ts={df['snapshot_ts'].iloc[0][:19]}")
        frames.append(df)

    rt = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(rt):,} rows before dedup")

    # Deduplicate on (trip_id, stop_id): keep the latest snapshot per pair
    # so that if multiple snapshots cover the same stop, we use the freshest.
    rt = rt.sort_values("snapshot_ts", ascending=False)
    rt = rt.drop_duplicates(subset=["trip_id", "stop_id"], keep="first")
    print(f"  After dedup on (trip_id, stop_id): {len(rt):,} rows")

    # Normalise IDs to strings to match GTFS category columns
    rt["trip_id"] = rt["trip_id"].astype(str)
    rt["stop_id"] = rt["stop_id"].astype(str)
    rt["route_id"] = rt["route_id"].astype(str)

    return rt


# ---------------------------------------------------------------------------
# Step 2 – Load GTFS + weather
# ---------------------------------------------------------------------------

def load_gtfs(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Convert category columns to str for the join
    df["trip_id"] = df["trip_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    print(f"  GTFS shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Unique trip_ids : {df['trip_id'].nunique():,}")
    print(f"  Unique stop_ids : {df['stop_id'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Step 3 – Merge
# ---------------------------------------------------------------------------

def merge_realtime(gtfs: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    # Only keep the RT columns we want to land in the final dataset
    rt_cols = [
        "trip_id", "stop_id",
        "snapshot_ts", "arrival_time", "arrival_delay",
        "departure_delay", "vehicle_id",
    ]
    rt_slim = rt[rt_cols].rename(columns={
        "snapshot_ts":      "rt_snapshot_ts",
        "arrival_time":     "rt_arrival_time",
        "arrival_delay":    "rt_arrival_delay",
        "departure_delay":  "rt_departure_delay",
        "vehicle_id":       "rt_vehicle_id",
    })

    # Left join — every GTFS row kept; RT columns NaN when no match
    merged = gtfs.merge(
        rt_slim,
        on=["trip_id", "stop_id"],
        how="left",
        validate="many_to_one",   # each (trip_id, stop_id) pair is unique in RT
    )
    return merged


# ---------------------------------------------------------------------------
# Step 4 – Delay feature engineering
# ---------------------------------------------------------------------------

def engineer_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # delay_minutes: NaN where no RT data was joined
    df["delay_minutes"] = df["rt_arrival_delay"] / 60.0

    # is_delayed: 0 for NaN rows (no realtime data = not confirmed delayed)
    df["is_delayed"] = (
        df["delay_minutes"]
        .gt(5)
        .fillna(False)
        .astype("Int8")
    )

    return df


# ---------------------------------------------------------------------------
# Step 5 – Stats report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    total          = len(df)
    has_delay_data = df["rt_arrival_delay"].notna().sum()
    pct_coverage   = has_delay_data / total * 100

    delayed        = df["is_delayed"].eq(1).sum()
    pct_delayed    = delayed / has_delay_data * 100 if has_delay_data > 0 else 0.0

    delay_valid    = df["delay_minutes"].dropna()
    avg_delay      = delay_valid.mean()
    median_delay   = delay_valid.median()
    max_delay      = delay_valid.max()

    print(f"\n  Total rows                     : {total:>9,}")
    print(f"  Rows with RT delay data        : {has_delay_data:>9,}  ({pct_coverage:.2f}%)")
    print(f"  Rows without RT data (NaN)     : {total - has_delay_data:>9,}")
    print(f"  is_delayed = 1 (> 5 min late)  : {delayed:>9,}  "
          f"({pct_delayed:.2f}% of matched rows)")
    print(f"  Average delay (matched rows)   : {avg_delay:>9.2f} min")
    print(f"  Median  delay (matched rows)   : {median_delay:>9.2f} min")
    print(f"  Max     delay (matched rows)   : {max_delay:>9.2f} min")

    # RT trip coverage
    rt_trips_in_gtfs  = df.loc[df["rt_arrival_delay"].notna(), "trip_id"].nunique()
    rt_trips_total    = df["trip_id"].nunique()
    print(f"\n  Unique GTFS trips total        : {rt_trips_total:>9,}")
    print(f"  Unique trips with RT data      : {rt_trips_in_gtfs:>9,}")

    # Top 5 most-delayed routes (by mean delay_minutes, min 10 RT rows)
    print(f"\n  Top 5 routes by mean delay (min 10 matched stop-updates):")
    route_delay = (
        df.dropna(subset=["delay_minutes"])
        .groupby("route_short_name", observed=True)
        .agg(
            mean_delay=("delay_minutes", "mean"),
            rt_rows=("delay_minutes", "count"),
            pct_delayed=("is_delayed", "mean"),
        )
        .query("rt_rows >= 10")
        .sort_values("mean_delay", ascending=False)
        .head(5)
    )
    if route_delay.empty:
        print("    (No routes with ≥10 matched RT rows — "
              "collect more snapshots during service hours)")
    else:
        print(f"    {'Route':<16} {'Mean delay (min)':>18} "
              f"{'RT rows':>9} {'% delayed':>10}")
        print(f"    {'-'*56}")
        for route, row in route_delay.iterrows():
            print(f"    {str(route):<16} {row['mean_delay']:>18.2f} "
                  f"{int(row['rt_rows']):>9,} {row['pct_delayed']*100:>9.1f}%")

    # Routes in RT snapshot but not in GTFS static
    rt_only = df.loc[df["rt_arrival_delay"].notna()].copy()
    # (these are already matched — show unmatched RT trip count separately)
    print(f"\n  is_rainy distribution (matched rows):")
    print(df.dropna(subset=["rt_arrival_delay"])["is_rainy"]
          .value_counts(dropna=False)
          .rename(index={0: "dry (0)", 1: "rainy (1)"})
          .to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("06_merge_realtime.py — Merge RT Delays into Final Dataset")
    print(SECTION)

    # 1. Load realtime
    print("\n[1] Loading realtime snapshots …")
    rt = load_realtime_snapshots(RAW)

    # 2. Load GTFS + weather
    print("\n[2] Loading GTFS + weather …")
    gtfs = load_gtfs(GTFS_FILE)

    # 3. Merge
    print("\n[3] Joining realtime → GTFS on (trip_id, stop_id) …")
    merged = merge_realtime(gtfs, rt)
    print(f"  Result: {merged.shape[0]:,} rows × {merged.shape[1]} cols")

    # 4. Feature engineering
    print("\n[4] Engineering delay features …")
    final = engineer_delay_features(merged)

    # 5. Report
    print(f"\n[5] Delay & coverage report")
    print_report(final)

    # 6. Final schema summary
    print(f"\n[6] Final schema ({final.shape[1]} columns):")
    schema = pd.DataFrame({
        "dtype":      final.dtypes.astype(str),
        "null_count": final.isnull().sum(),
        "null_%":     (final.isnull().sum() / len(final) * 100).round(2),
    })
    print(schema.to_string())

    # 7. Sample rows with delay data
    sample = final.dropna(subset=["rt_arrival_delay"])
    if not sample.empty:
        print(f"\n[7] Sample rows with RT delay data (5 rows):")
        show_cols = [
            "trip_id", "route_short_name", "stop_id", "stop_sequence",
            "arrival_sec", "rt_arrival_delay", "delay_minutes", "is_delayed",
            "rt_vehicle_id", "rt_snapshot_ts",
        ]
        show_cols = [c for c in show_cols if c in final.columns]
        print(sample[show_cols].head(5).to_string(index=False))
    else:
        print("\n[7] No rows with RT delay data in this snapshot "
              "(run during service hours for live delays).")

    # 8. Save
    print(f"\n[8] Saving outputs …")
    final.to_parquet(OUT_PARQUET, index=False)
    print(f"  Parquet → {OUT_PARQUET}  ({final.shape[0]:,} rows × {final.shape[1]} cols)")

    # CSV export — timedelta columns can't be written to CSV natively; convert to seconds
    csv_df = final.copy()
    for col in csv_df.select_dtypes(include=["timedelta64[ns]"]).columns:
        csv_df[col] = csv_df[col].dt.total_seconds()
    csv_df.to_csv(OUT_CSV, index=False)
    print(f"  CSV    → {OUT_CSV}  (timedelta columns serialised as total seconds)")

    print(f"\n{SECTION}")
    print("Done. Final dataset saved to data/cleaned/vta_final.*")
    print(SECTION)


if __name__ == "__main__":
    main()
