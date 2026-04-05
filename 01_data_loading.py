"""
01_data_loading.py
------------------
Load VTA GTFS static data, filter to VTA bus routes only
(excludes School trippers and the CTBUS / Caltrain bus-bridge),
merge stop_times + trips + routes into one flat DataFrame,
and print shape + sample rows.

Folder layout expected:
  GROUP_PROJECT/
    data/
      raw/          <- GTFS txt files live here
      cleaned/      <- write derived outputs here
    01_data_loading.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------

def load_gtfs_tables() -> dict[str, pd.DataFrame]:
    """Read all six GTFS text files as string-typed DataFrames."""
    files = [
        "stop_times",
        "trips",
        "routes",
        "stops",
        "calendar",
        "calendar_dates",
    ]
    tables = {}
    for name in files:
        path = RAW / f"{name}.txt"
        tables[name] = pd.read_csv(path, dtype=str, low_memory=False)
        print(f"  Loaded {name:20s}  {tables[name].shape[0]:>7,} rows  "
              f"{tables[name].shape[1]} cols")
    return tables

# ---------------------------------------------------------------------------
# 2. Filter to VTA bus routes
# ---------------------------------------------------------------------------

def filter_vta_bus_routes(routes: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only VTA-operated bus routes (route_type == 3).
    Exclude:
      - School trippers  (route_id starts with 'School')
      - CTBUS            (Caltrain Tamien-Diridon bus bridge, route_id == 'CTBUS')

    Note: route 89 ('California Ave Caltrain - Palo Alto VA Hosp') is a regular
    VTA bus and is intentionally kept; only the literal CTBUS route is removed.
    """
    rt = routes.copy()
    rt["route_type"] = pd.to_numeric(rt["route_type"], errors="coerce")

    is_bus = rt["route_type"].eq(3)
    not_school = ~rt["route_id"].str.startswith("School", na=False)
    not_ctbus = rt["route_id"].ne("CTBUS")

    filtered = rt.loc[is_bus & not_school & not_ctbus].copy()

    removed = len(routes) - len(filtered)
    print(f"\n  routes total : {len(routes):>4}")
    print(f"  excluded     : {removed:>4}  (School trippers + CTBUS/Caltrain)")
    print(f"  VTA bus only : {len(filtered):>4}")
    return filtered

# ---------------------------------------------------------------------------
# 3. Merge stop_times + trips + routes
# ---------------------------------------------------------------------------

def build_flat_dataframe(
    stop_times: pd.DataFrame,
    trips: pd.DataFrame,
    routes_bus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join stop_times  many:1  trips  many:1  routes
    keeping only rows that belong to VTA bus trips.
    """
    # trips that belong to a VTA bus route
    trips_bus = trips.loc[trips["route_id"].isin(routes_bus["route_id"])].copy()

    # stop_times for those trips only
    st_bus = stop_times.loc[stop_times["trip_id"].isin(trips_bus["trip_id"])].copy()

    merged = (
        st_bus
        .merge(trips_bus, on="trip_id", how="inner", validate="many_to_one")
        .merge(routes_bus, on="route_id", how="inner", validate="many_to_one",
               suffixes=("_trip", "_route"))
    )
    return merged

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("VTA GTFS Data Loading")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading raw GTFS tables from:", RAW)
    tables = load_gtfs_tables()

    # 2. Filter routes
    print("\n[2] Filtering to VTA bus routes …")
    routes_bus = filter_vta_bus_routes(tables["routes"])

    # 3. Merge
    print("\n[3] Merging stop_times → trips → routes …")
    flat = build_flat_dataframe(
        tables["stop_times"],
        tables["trips"],
        routes_bus,
    )

    # 4. Report
    print(f"\n[4] Merged DataFrame shape: {flat.shape}")
    print(f"    Columns: {list(flat.columns)}\n")
    print("Sample rows (first 5):")
    display_cols = [
        "trip_id", "route_id", "route_short_name", "route_long_name",
        "stop_id", "stop_sequence", "arrival_time", "departure_time",
        "trip_headsign", "direction_id",
    ]
    display_cols = [c for c in display_cols if c in flat.columns]
    print(flat[display_cols].head(5).to_string(index=False))

    # 5. Persist cleaned output
    out_path = CLEANED / "stop_times_merged.parquet"
    flat.to_parquet(out_path, index=False)
    print(f"\n[5] Cleaned merged file saved to: {out_path}")


if __name__ == "__main__":
    main()
