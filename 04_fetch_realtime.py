"""
04_fetch_realtime.py
--------------------
Fetch a GTFS-RT TripUpdates snapshot from the 511.org API for VTA (agency=SC),
parse the protobuf binary, extract stop-time-level delay fields, and save as CSV.

Output
------
  data/raw/realtime_snapshot_<N>.csv   (auto-increments so reruns don't overwrite)

Columns
-------
  snapshot_ts       UTC timestamp when this snapshot was captured (ISO-8601)
  feed_ts           POSIX timestamp from the feed header
  entity_id         GTFS-RT entity id
  trip_id           trip identifier (matches GTFS static trips.txt)
  route_id          route identifier (matches GTFS static routes.txt)
  vehicle_id        vehicle label (if provided by the feed)
  stop_id           stop identifier
  stop_sequence     stop sequence number within the trip
  arrival_time      absolute arrival time (POSIX, if provided)
  arrival_delay     arrival delay in seconds  (+ = late, - = early, None = not provided)
  departure_time    absolute departure time (POSIX, if provided)
  departure_delay   departure delay in seconds (same sign convention)
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
    "https://api.511.org/transit/tripupdates"
    "?api_key=a2136af1-e13e-4dc0-a575-be031a265b41"
    "&agency=SC"
)
HEADERS = {"Accept": "application/x-protobuf"}
TIMEOUT = 30  # seconds

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Step 1 – Fetch
# ---------------------------------------------------------------------------

def fetch_feed(url: str) -> bytes:
    print(f"  GET {url[:80]}…")
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    print(f"  HTTP {resp.status_code}  |  {len(resp.content):,} bytes  "
          f"|  Content-Type: {resp.headers.get('content-type', 'unknown')}")
    return resp.content


# ---------------------------------------------------------------------------
# Step 2 – Parse protobuf
# ---------------------------------------------------------------------------

def parse_feed(raw: bytes) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(raw)
    return feed


# ---------------------------------------------------------------------------
# Step 3 – Extract rows
# ---------------------------------------------------------------------------

def extract_rows(
    feed: gtfs_realtime_pb2.FeedMessage,
    snapshot_ts: datetime,
) -> list[dict]:
    """
    Walk every TripUpdate entity and every StopTimeUpdate within it,
    extracting delay/time fields with safe None defaults when absent.
    """
    rows: list[dict] = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu      = entity.trip_update
        trip_id  = tu.trip.trip_id  or None
        route_id = tu.trip.route_id or None
        vehicle_id = (
            tu.vehicle.label or tu.vehicle.id
            if tu.HasField("vehicle") else None
        )

        for stu in tu.stop_time_update:
            arrival_delay    = None
            arrival_time_pos = None
            if stu.HasField("arrival"):
                arrival_delay    = stu.arrival.delay if stu.arrival.delay != 0 \
                                   or stu.arrival.HasField != int else stu.arrival.delay
                # protobuf int fields default to 0, so always capture as-is
                arrival_delay    = stu.arrival.delay
                arrival_time_pos = stu.arrival.time or None

            departure_delay    = None
            departure_time_pos = None
            if stu.HasField("departure"):
                departure_delay    = stu.departure.delay
                departure_time_pos = stu.departure.time or None

            rows.append({
                "snapshot_ts":      snapshot_ts.isoformat(),
                "feed_ts":          feed.header.timestamp,
                "entity_id":        entity.id,
                "trip_id":          trip_id,
                "route_id":         route_id,
                "vehicle_id":       vehicle_id,
                "stop_id":          stu.stop_id or None,
                "stop_sequence":    stu.stop_sequence if stu.stop_sequence else None,
                "arrival_time":     arrival_time_pos,
                "arrival_delay":    arrival_delay,
                "departure_time":   departure_time_pos,
                "departure_delay":  departure_delay,
            })

    return rows


# ---------------------------------------------------------------------------
# Step 4 – Auto-increment output filename
# ---------------------------------------------------------------------------

def next_snapshot_path(raw_dir: Path) -> Path:
    existing = sorted(raw_dir.glob("realtime_snapshot_*.csv"))
    if not existing:
        return raw_dir / "realtime_snapshot_1.csv"
    last_n = int(existing[-1].stem.split("_")[-1])
    return raw_dir / f"realtime_snapshot_{last_n + 1}.csv"


# ---------------------------------------------------------------------------
# Step 5 – Summary report
# ---------------------------------------------------------------------------

def print_report(feed: gtfs_realtime_pb2.FeedMessage, df: pd.DataFrame,
                 out_path: Path) -> None:
    trip_updates    = sum(1 for e in feed.entity if e.HasField("trip_update"))
    stop_updates    = len(df)
    with_arr_delay  = df["arrival_delay"].notna().sum()
    with_dep_delay  = df["departure_delay"].notna().sum()
    routes_live     = df["route_id"].nunique()
    trips_live      = df["trip_id"].nunique()

    print(f"\n{SECTION}")
    print("GTFS-RT Snapshot Summary")
    print(SECTION)
    print(f"  Feed timestamp        : {feed.header.timestamp}  "
          f"({datetime.fromtimestamp(feed.header.timestamp, tz=timezone.utc).isoformat()})")
    print(f"  TripUpdate entities   : {trip_updates:>6,}")
    print(f"  StopTimeUpdate rows   : {stop_updates:>6,}")
    print(f"  Unique trips          : {trips_live:>6,}")
    print(f"  Unique routes         : {routes_live:>6,}")
    print(f"  With arrival delay    : {with_arr_delay:>6,}")
    print(f"  With departure delay  : {with_dep_delay:>6,}")

    if with_arr_delay > 0:
        print(f"\n  Arrival delay stats (seconds):")
        print(f"    mean  : {df['arrival_delay'].mean():>8.1f}")
        print(f"    median: {df['arrival_delay'].median():>8.1f}")
        print(f"    min   : {df['arrival_delay'].min():>8.0f}")
        print(f"    max   : {df['arrival_delay'].max():>8.0f}")
        late = (df["arrival_delay"] > 0).sum()
        early = (df["arrival_delay"] < 0).sum()
        on_time = (df["arrival_delay"] == 0).sum()
        print(f"    late (>0s)   : {late:>6,}")
        print(f"    on-time (=0) : {on_time:>6,}")
        print(f"    early (<0s)  : {early:>6,}")

    print(f"\n  Top 5 routes in snapshot (by stop updates):")
    top_routes = df["route_id"].value_counts().head(5)
    for route, cnt in top_routes.items():
        print(f"    Route {str(route):<12} {cnt:>4,} stop updates")

    print(f"\n  Saved → {out_path}")
    print(SECTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("04_fetch_realtime.py — VTA GTFS-RT TripUpdates")
    print(SECTION)

    snapshot_ts = datetime.now(tz=timezone.utc)

    # 1. Fetch
    print("\n[1] Fetching feed …")
    raw_bytes = fetch_feed(API_URL)

    # 2. Parse
    print("\n[2] Parsing protobuf …")
    feed = parse_feed(raw_bytes)
    print(f"  Entities in feed: {len(feed.entity)}")

    # 3. Extract
    print("\n[3] Extracting stop-time updates …")
    rows = extract_rows(feed, snapshot_ts)
    if not rows:
        print("  WARNING: No TripUpdate entities found in feed. Exiting.")
        sys.exit(1)
    print(f"  Extracted {len(rows):,} stop-time update rows "
          f"from {sum(1 for e in feed.entity if e.HasField('trip_update')):,} trip updates")

    # 4. Build DataFrame and save
    print("\n[4] Saving CSV …")
    df = pd.DataFrame(rows)
    out_path = next_snapshot_path(RAW)
    df.to_csv(out_path, index=False)
    print(f"  Written {len(df):,} rows × {len(df.columns)} cols → {out_path.name}")

    # 5. Report
    print_report(feed, df, out_path)

    # 6. Preview
    print("\nSample rows (first 5):")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
