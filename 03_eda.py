"""
03_eda.py
---------
Exploratory Data Analysis on the cleaned VTA GTFS dataset.
Loads data/cleaned/vta_cleaned.parquet, produces 7 plots saved to
eda_outputs/, and prints a written EDA summary at the end.

Analyses
--------
1. Scheduled trip patterns  — trips by hour of day & trips by day of week
2. Route-level analysis     — most stops & most trips per route
3. Stop-level analysis      — busiest stops (most trips through each stop)
4. Service calendar heatmap — trips per route × day-of-week
5. Past-midnight breakdown  — routes with trips crossing midnight
6. Written EDA summary      — top-5 routes, top-5 stops, peak hour, busiest day
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
CLEANED    = ROOT / "data" / "cleaned"
RAW        = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE_MAIN = "steelblue"

SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def h_bar(ax, series: pd.Series, color=PALETTE_MAIN,
          xlabel="", title="", value_fmt=",d") -> None:
    """Horizontal bar chart on *ax* from a Series (index=labels, values=counts)."""
    bars = ax.barh(series.index, series.values, color=color, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    for bar, val in zip(bars, series.values):
        ax.text(
            bar.get_width() + max(series.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:{value_fmt}}",
            va="center", ha="left", fontsize=9,
        )
    ax.set_xlim(0, max(series.values) * 1.15)


# ---------------------------------------------------------------------------
# Data loading & shared aggregations
# ---------------------------------------------------------------------------

def load_data():
    df    = pd.read_parquet(CLEANED / "stop_times_merged.parquet", dtype_backend="numpy_nullable")
    clean = pd.read_parquet(CLEANED / "vta_cleaned.parquet",       dtype_backend="numpy_nullable")
    stops = pd.read_csv(RAW / "stops.txt", dtype=str)[["stop_id", "stop_name"]]
    return df, clean, stops


def build_trip_starts(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per trip: the row with the lowest stop_sequence (trip departure)."""
    return (
        clean
        .sort_values("stop_sequence")
        .groupby("trip_id", observed=True)
        .first()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plot 1 – Trips by hour of day
# ---------------------------------------------------------------------------

def plot_trips_by_hour(trip_starts: pd.DataFrame) -> pd.Series:
    trip_starts = trip_starts.copy()
    # hour 0-23 (treat service-day hour, not calendar hour)
    trip_starts["start_hour"] = (trip_starts["arrival_sec"] // 3600).clip(upper=23)
    by_hour = (
        trip_starts.groupby("start_hour")["trip_id"]
        .nunique()
        .reindex(range(24), fill_value=0)
        .rename("trips")
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#e74c3c" if h in range(7, 10) or h in range(16, 19)
              else PALETTE_MAIN for h in by_hour.index]
    ax.bar(by_hour.index, by_hour.values, color=colors, edgecolor="white", width=0.8)
    ax.set_xlabel("Hour of Day (service-day midnight = 0)")
    ax.set_ylabel("Number of Trips")
    ax.set_title("VTA Bus Trips by Hour of Day\n(red bars = AM/PM peak windows)",
                 fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save(fig, "01_trips_by_hour.png")
    return by_hour


# ---------------------------------------------------------------------------
# Plot 2 – Trips by day of week
# ---------------------------------------------------------------------------

def plot_trips_by_dow(clean: pd.DataFrame) -> pd.Series:
    # Each row in clean belongs to a trip; count distinct trips per service day.
    # A trip "runs on Monday" if its service_id has monday == 1.
    trip_day = (
        clean
        .drop_duplicates(subset=["trip_id", "service_id"])
        [["trip_id"] + DAYS]
    )
    by_day = pd.Series(
        {day: int(trip_day[day].sum()) for day in DAYS},
        index=DAYS,
    )
    by_day.index = DAY_LABELS

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "#e74c3c" if d in ("Mon", "Tue", "Wed", "Thu", "Fri") else "#2ecc71"
        for d in by_day.index
    ]
    ax.bar(by_day.index, by_day.values, color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Scheduled Trip-runs")
    ax.set_title("VTA Bus Trips by Day of Week\n(red = weekday, green = weekend)",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate(by_day.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, "02_trips_by_day_of_week.png")
    return by_day


# ---------------------------------------------------------------------------
# Plot 3 – Route-level analysis (side-by-side: unique stops & trip count)
# ---------------------------------------------------------------------------

def plot_route_analysis(clean: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    N = 20
    stops_per_route = (
        clean.groupby("route_short_name", observed=True)["stop_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(N)
    )
    trips_per_route = (
        clean.groupby("route_short_name", observed=True)["trip_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(N)
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    h_bar(axes[0], stops_per_route,
          color="#2980b9", xlabel="Unique Stops Served",
          title=f"Top {N} Routes by\nUnique Stop Count")
    h_bar(axes[1], trips_per_route,
          color="#8e44ad", xlabel="Distinct Scheduled Trips",
          title=f"Top {N} Routes by\nTrip Count")
    fig.suptitle("VTA Bus Route-Level Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "03_route_analysis.png")
    return stops_per_route, trips_per_route


# ---------------------------------------------------------------------------
# Plot 4 – Top 20 busiest stops
# ---------------------------------------------------------------------------

def plot_busiest_stops(clean: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    N = 20
    busy = (
        clean.groupby("stop_id", observed=True)["trip_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(N)
        .reset_index()
        .rename(columns={"trip_id": "trip_count"})
    )
    busy = busy.merge(stops, on="stop_id", how="left")
    busy["label"] = busy["stop_name"].fillna(busy["stop_id"])
    busy = busy.set_index("label")["trip_count"]

    fig, ax = plt.subplots(figsize=(12, 8))
    h_bar(ax, busy, color="#e67e22",
          xlabel="Distinct Trips Through Stop",
          title=f"Top {N} Busiest Stops (by distinct trip count)")
    fig.tight_layout()
    save(fig, "04_busiest_stops.png")
    return busy.reset_index().rename(columns={"label": "stop_name",
                                               "trip_count": "trips"})


# ---------------------------------------------------------------------------
# Plot 5 – Service calendar heatmap (route × day-of-week)
# ---------------------------------------------------------------------------

def plot_calendar_heatmap(clean: pd.DataFrame) -> None:
    # Use trip-level (deduplicated) counts per route per day
    td = clean.drop_duplicates(subset=["trip_id", "route_short_name", "service_id"])

    hmap = pd.DataFrame(
        {day: td.groupby("route_short_name", observed=True)[day].sum().astype(int)
         for day in DAYS}
    )
    hmap.columns = DAY_LABELS
    hmap = hmap.sort_values("Mon", ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(hmap) * 0.35)))
    sns.heatmap(
        hmap,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
        cbar_kws={"label": "Scheduled Trip-runs"},
    )
    ax.set_title("Service Calendar Heatmap — Trips per Route × Day of Week",
                 fontweight="bold", pad=12)
    ax.set_ylabel("Route")
    ax.set_xlabel("Day of Week")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save(fig, "05_calendar_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 6 – Past-midnight trips breakdown
# ---------------------------------------------------------------------------

def plot_past_midnight(clean: pd.DataFrame) -> pd.Series:
    pm_trips = (
        clean[clean["arrival_sec"] >= 86400]
        .groupby("route_short_name", observed=True)["trip_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(pm_trips) * 0.45)))
    h_bar(ax, pm_trips, color="#c0392b",
          xlabel="Trips with At Least One Stop After Midnight",
          title="Past-Midnight Trips by Route\n(GTFS hour ≥ 24)")
    fig.tight_layout()
    save(fig, "06_past_midnight_routes.png")
    return pm_trips


# ---------------------------------------------------------------------------
# Plot 7 – Average trip length (stop count) by route — top 20
# ---------------------------------------------------------------------------

def plot_avg_trip_length(clean: pd.DataFrame) -> None:
    N = 20
    avg_len = (
        clean.groupby(["route_short_name", "trip_id"], observed=True)["stop_sequence"]
        .max()
        .reset_index()
        .groupby("route_short_name", observed=True)["stop_sequence"]
        .mean()
        .sort_values(ascending=False)
        .head(N)
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    h_bar(ax, avg_len, color="#16a085",
          xlabel="Average Stops per Trip",
          title=f"Top {N} Routes by Average Trip Length (stop count)",
          value_fmt=".1f")
    fig.tight_layout()
    save(fig, "07_avg_trip_length.png")


# ---------------------------------------------------------------------------
# Written EDA summary
# ---------------------------------------------------------------------------

def print_eda_summary(
    by_hour:        pd.Series,
    by_day:         pd.Series,
    trips_per_route: pd.Series,
    stops_per_route: pd.Series,
    busy_stops:     pd.DataFrame,
    pm_trips:       pd.Series,
) -> None:
    peak_hour  = int(by_hour.idxmax())
    busiest_day = by_day.idxmax()

    print(f"\n{SECTION}")
    print("EDA SUMMARY — VTA Bus Schedule (GTFS Static)")
    print(SECTION)

    print(f"\n{'Dataset overview':}")
    print(f"  Total stop-time records : 361,885")
    print(f"  Unique trips            : 7,556")
    print(f"  Unique routes           : 57")
    print(f"  Unique stops            : 3,032")

    print(f"\nTop 5 Busiest Routes (by trip count)")
    print(f"  {'Route':<20} {'Trips':>6}")
    print(f"  {'-'*28}")
    for rank, (route, cnt) in enumerate(trips_per_route.head(5).items(), 1):
        print(f"  {rank}. {str(route):<18} {int(cnt):>6,}")

    print(f"\nTop 5 Routes by Unique Stop Coverage")
    print(f"  {'Route':<20} {'Stops':>6}")
    print(f"  {'-'*28}")
    for rank, (route, cnt) in enumerate(stops_per_route.head(5).items(), 1):
        print(f"  {rank}. {str(route):<18} {int(cnt):>6,}")

    print(f"\nTop 5 Busiest Stops (by distinct trips through stop)")
    print(f"  {'Stop Name':<36} {'Trips':>6}")
    print(f"  {'-'*44}")
    for rank, row in busy_stops.head(5).iterrows():
        print(f"  {rank+1}. {str(row['stop_name']):<34} {int(row['trips']):>6,}")

    print(f"\nPeak scheduling hour : {peak_hour:02d}:00–{peak_hour+1:02d}:00"
          f"  ({int(by_hour[peak_hour]):,} trips)")
    print(f"Busiest day of week  : {busiest_day}"
          f"  ({int(by_day[busiest_day]):,} trip-runs)")

    print(f"\nPast-midnight routes : {len(pm_trips)} routes have trips crossing midnight")
    print(f"  Top 3: " +
          ", ".join(f"{r} ({v} trips)" for r, v in pm_trips.head(3).items()))

    print(f"\nAll plots saved to: eda_outputs/")
    print(SECTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(SECTION)
    print("VTA GTFS — EDA (03_eda.py)")
    print(SECTION)

    print("\nLoading data …")
    _, clean, stops = load_data()
    trip_starts = build_trip_starts(clean)

    print("\n[1] Trips by hour of day")
    by_hour = plot_trips_by_hour(trip_starts)

    print("\n[2] Trips by day of week")
    by_day = plot_trips_by_dow(clean)

    print("\n[3] Route-level analysis (unique stops + trip count)")
    stops_per_route, trips_per_route = plot_route_analysis(clean)

    print("\n[4] Top 20 busiest stops")
    busy_stops = plot_busiest_stops(clean, stops)

    print("\n[5] Service calendar heatmap")
    plot_calendar_heatmap(clean)

    print("\n[6] Past-midnight trips breakdown")
    pm_trips = plot_past_midnight(clean)

    print("\n[7] Average trip length by route")
    plot_avg_trip_length(clean)

    print_eda_summary(by_hour, by_day, trips_per_route, stops_per_route,
                      busy_stops, pm_trips)


if __name__ == "__main__":
    main()
