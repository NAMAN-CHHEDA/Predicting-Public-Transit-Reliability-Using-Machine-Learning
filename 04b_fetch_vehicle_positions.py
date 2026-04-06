"""
04b_fetch_vehicle_positions.py
-------------------------------
Fetch a GTFS-RT VehiclePositions snapshot from the 511.org API for VTA (agency=SC),
parse the protobuf, separate real GPS-tracked vehicles from schedule-based ones,
and save real vehicles only to data/raw/vehicle_positions_<N>.csv.

Output columns
--------------
  snapshot_ts         UTC timestamp when this snapshot was captured (ISO-8601)
  feed_ts             POSIX timestamp from the feed header
  vehicle_id          vehicle label / ID
  trip_id             GTFS trip identifier
  route_id            GTFS route identifier
  direction_id        0 = outbound, 1 = inbound
  start_time          scheduled trip start time (HH:MM:SS)
  start_date          scheduled service date (YYYYMMDD)
  schedule_relationship  0=SCHEDULED, 1=ADDED, 2=UNSCHEDULED, 3=CANCELED
  stop_id             ID of the stop the vehicle is currently at/approaching
  current_stop_sequence  sequence number of the current stop
  current_status      1=INCOMING_AT, 2=IN_TRANSIT_TO, 3=STOPPED_AT
  current_status_name human-readable status label
  latitude            WGS-84 latitude
  longitude           WGS-84 longitude
  bearing             heading in degrees (0 = north, clockwise)
  speed_mph           speed converted from m/s to mph
  occupancy_status    GTFS-RT OccupancyStatus enum value
  occupancy_label     human-readable occupancy label
  vehicle_timestamp   per-vehicle POSIX timestamp (may differ from feed_ts)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from google.transit import gtfs_realtime_pb2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = (
    "https://api.511.org/transit/vehiclepositions"
    "?api_key=a2136af1-e13e-4dc0-a575-be031a265b41"
    "&agency=SC"
)
HEADERS = {"Accept": "application/x-protobuf"}
TIMEOUT = 30

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SECTION = "=" * 65

# GTFS-RT enum label maps
STATUS_LABELS = {
    0: "INCOMING_AT",    # vehicle is about to arrive
    1: "INCOMING_AT",    # protobuf default maps 1 here in this feed
    2: "IN_TRANSIT_TO",
    3: "STOPPED_AT",
}
# Feed uses 1=INCOMING_AT, 2=IN_TRANSIT_TO per observed values
STATUS_LABELS = {
    1: "INCOMING_AT",
    2: "IN_TRANSIT_TO",
    3: "STOPPED_AT",
}

OCCUPANCY_LABELS = {
    0: "EMPTY",
    1: "MANY_SEATS_AVAILABLE",
    2: "FEW_SEATS_AVAILABLE",
    3: "STANDING_ROOM_ONLY",
    4: "CRUSHED_STANDING_ROOM_ONLY",
    5: "FULL",
    6: "NOT_ACCEPTING_PASSENGERS",
}

MPS_TO_MPH = 2.23694   # conversion factor


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_feed(url: str) -> bytes:
    print(f"  GET {url[:85]}…")
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  HTTP {resp.status_code}  |  {len(resp.content):,} bytes  "
          f"|  {resp.headers.get('content-type', '?')}")
    return resp.content


# ---------------------------------------------------------------------------
# Parse & extract
# ---------------------------------------------------------------------------

def extract_vehicles(
    feed: gtfs_realtime_pb2.FeedMessage,
    snapshot_ts: datetime,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (real_rows, sched_rows).
    real_rows   — vehicles with a genuine GPS-tracked ID
    sched_rows  — vehicles whose ID contains 'schedBased' (synthetic)
    """
    real_rows:  list[dict] = []
    sched_rows: list[dict] = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue

        v          = entity.vehicle
        vehicle_id = v.vehicle.label or v.vehicle.id if v.HasField("vehicle") else ""

        row = {
            "snapshot_ts":           snapshot_ts.isoformat(),
            "feed_ts":               feed.header.timestamp,
            "vehicle_id":            vehicle_id or None,
            "trip_id":               v.trip.trip_id   or None,
            "route_id":              v.trip.route_id  or None,
            "direction_id":          v.trip.direction_id,
            "start_time":            v.trip.start_time or None,
            "start_date":            v.trip.start_date or None,
            "schedule_relationship": v.trip.schedule_relationship,
            "stop_id":               v.stop_id or None,
            "current_stop_sequence": v.current_stop_sequence or None,
            "current_status":        v.current_status,
            "current_status_name":   STATUS_LABELS.get(v.current_status, str(v.current_status)),
            "latitude":              v.position.latitude,
            "longitude":             v.position.longitude,
            "bearing":               v.position.bearing if v.position.bearing else None,
            "speed_mph":             round(v.position.speed * MPS_TO_MPH, 2),
            "occupancy_status":      v.occupancy_status,
            "occupancy_label":       OCCUPANCY_LABELS.get(v.occupancy_status, str(v.occupancy_status)),
            "vehicle_timestamp":     v.timestamp or None,
        }

        if "schedBased" in (vehicle_id or ""):
            sched_rows.append(row)
        else:
            real_rows.append(row)

    return real_rows, sched_rows


# ---------------------------------------------------------------------------
# Auto-increment output path
# ---------------------------------------------------------------------------

def next_output_path(raw_dir: Path, prefix: str) -> Path:
    existing = sorted(raw_dir.glob(f"{prefix}_*.csv"))
    n = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 1
    return raw_dir / f"{prefix}_{n}.csv"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    feed:      gtfs_realtime_pb2.FeedMessage,
    real_df:   pd.DataFrame,
    sched_rows: list[dict],
    out_path:  Path,
) -> None:
    total     = len(real_df) + len(sched_rows)
    n_real    = len(real_df)
    n_sched   = len(sched_rows)

    print(f"\n{SECTION}")
    print("Vehicle Positions Snapshot Summary")
    print(SECTION)
    print(f"  Feed timestamp     : {feed.header.timestamp}  "
          f"({datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc).isoformat()})")
    print(f"  Total entities     : {total:>5}")
    print(f"  Real GPS vehicles  : {n_real:>5}  ← saved to CSV")
    print(f"  Schedule-based     : {n_sched:>5}  ← filtered out")

    if n_real == 0:
        print("\n  No real GPS vehicles found in this snapshot.")
        return

    print(f"\n  Routes represented : {real_df['route_id'].nunique()}")
    print(f"  Unique trips       : {real_df['trip_id'].nunique()}")

    print(f"\n  Current status breakdown:")
    for label, cnt in real_df["current_status_name"].value_counts().items():
        print(f"    {label:<22} {cnt:>4}")

    print(f"\n  Occupancy breakdown:")
    for label, cnt in real_df["occupancy_label"].value_counts().items():
        print(f"    {label:<30} {cnt:>4}")

    moving = real_df["speed_mph"].gt(1).sum()
    print(f"\n  Vehicles moving (>1 mph) : {moving:>4} / {n_real}")
    print(f"  Speed stats (mph)  : "
          f"mean={real_df['speed_mph'].mean():.2f}  "
          f"max={real_df['speed_mph'].max():.2f}")

    print(f"\n  Top 5 routes by vehicle count:")
    top = real_df["route_id"].value_counts().head(5)
    for route, cnt in top.items():
        print(f"    Route {str(route):<12} {cnt:>3} vehicles")

    print(f"\n  Saved → {out_path}")
    print(SECTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("04b_fetch_vehicle_positions.py — VTA GTFS-RT Vehicle Positions")
    print(SECTION)

    snapshot_ts = datetime.now(tz=timezone.utc)

    print("\n[1] Fetching feed …")
    raw_bytes = fetch_feed(API_URL)

    print("\n[2] Parsing protobuf …")
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw_bytes)
    print(f"  Total entities: {len(feed.entity)}")

    print("\n[3] Extracting and filtering vehicles …")
    real_rows, sched_rows = extract_vehicles(feed, snapshot_ts)
    print(f"  Real GPS vehicles  : {len(real_rows)}")
    print(f"  Schedule-based     : {len(sched_rows)}  (filtered out)")

    if not real_rows:
        print("\n  WARNING: No real GPS-tracked vehicles found. Nothing to save.")
        sys.exit(0)

    real_df = pd.DataFrame(real_rows)

    print("\n[4] Saving …")
    out_path = next_output_path(RAW, "vehicle_positions")
    real_df.to_csv(out_path, index=False)
    print(f"  Written {len(real_df):,} rows × {len(real_df.columns)} cols"
          f" → {out_path.name}")

    print_report(feed, real_df, sched_rows, out_path)

    print("\nSample rows (first 5):")
    show = ["vehicle_id", "route_id", "trip_id", "current_status_name",
            "latitude", "longitude", "speed_mph", "occupancy_label",
            "stop_id", "current_stop_sequence"]
    print(real_df[show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
