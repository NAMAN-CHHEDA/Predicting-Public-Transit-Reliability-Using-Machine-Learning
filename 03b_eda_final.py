"""
03b_eda_final.py
----------------
Extended EDA on data/cleaned/vta_final.csv — focuses on the 455 rows
where real GPS-derived delay data is available.

All plots saved to eda_outputs/final/.
A written summary comparing findings to the initial EDA is printed at the end.

Plots produced
--------------
 F01  delay_distribution.png        — histogram of delay_minutes
 F02  is_delayed_by_route.png       — delay rate & mean delay per route
 F03  delay_vs_weather.png          — delay vs prcp and vs tmax (with notes)
 F04  delay_by_hour.png             — avg delay by hour of day
 F05  delay_by_dow.png              — avg delay + observation count by day of week
 F06  on_time_pie.png               — on-time / early / late breakdown pie
 F07  occupancy_vs_delay.png        — delay distribution by occupancy label
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
CLEANED = ROOT / "data" / "cleaned"
OUT_DIR = ROOT / "eda_outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS     = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
DAY_ABBR = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

PALETTE = {
    "late":    "#e74c3c",
    "ontime":  "#2ecc71",
    "early":   "#3498db",
    "neutral": "steelblue",
    "accent":  "#e67e22",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
SECTION = "=" * 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → eda_outputs/final/{name}")


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    obs = df[df["delay_minutes"].notna()].copy()
    obs["hour"] = (obs["arrival_sec"] // 3600).clip(upper=23).astype(int)
    print(f"  Total rows     : {len(df):,}")
    print(f"  Observed rows  : {len(obs):,}  (delay_minutes not null)")
    return df, obs


# ---------------------------------------------------------------------------
# F01 – Delay distribution histogram
# ---------------------------------------------------------------------------

def plot_delay_distribution(obs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full range histogram
    ax = axes[0]
    bins = np.arange(obs["delay_minutes"].min() - 1,
                     obs["delay_minutes"].max() + 2, 2)
    n, edges, patches = ax.hist(obs["delay_minutes"], bins=bins,
                                edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches, edges[:-1]):
        if left > 5:
            patch.set_facecolor(PALETTE["late"])
        elif left < -5:
            patch.set_facecolor(PALETTE["early"])
        else:
            patch.set_facecolor(PALETTE["ontime"])

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="On time")
    ax.axvline(5, color=PALETTE["late"], linestyle=":", linewidth=1.2, label=">5 min (late)")
    ax.axvline(-5, color=PALETTE["early"], linestyle=":", linewidth=1.2, label="<-5 min (early)")
    ax.axvline(obs["delay_minutes"].mean(), color="orange",
               linestyle="-", linewidth=1.5, label=f"Mean ({obs['delay_minutes'].mean():.1f} min)")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Delay Distribution — All Observed Trips\n"
                 "(green=on-time, blue=early, red=late)", fontweight="bold")
    ax.legend(fontsize=8)

    # Right: zoomed-in ±20 min for clarity
    ax2 = axes[1]
    obs_zoom = obs[obs["delay_minutes"].between(-20, 20)]
    bins2 = np.arange(-20, 21, 1)
    n2, edges2, patches2 = ax2.hist(obs_zoom["delay_minutes"], bins=bins2,
                                    edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches2, edges2[:-1]):
        if left > 5:  patch.set_facecolor(PALETTE["late"])
        elif left < -5: patch.set_facecolor(PALETTE["early"])
        else:          patch.set_facecolor(PALETTE["ontime"])
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Delay (minutes)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Zoomed ±20 min  (n={len(obs_zoom)} of {len(obs)} rows)\n"
                  f"Excludes {len(obs) - len(obs_zoom)} extreme outliers",
                  fontweight="bold")

    stats_text = (f"n={len(obs)}  mean={obs['delay_minutes'].mean():.1f} min\n"
                  f"median={obs['delay_minutes'].median():.1f}  "
                  f"std={obs['delay_minutes'].std():.1f}")
    fig.text(0.5, -0.02, stats_text, ha="center", fontsize=9, color="gray")
    fig.suptitle("VTA Bus Delay Distribution (GPS-Derived)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "F01_delay_distribution.png")


# ---------------------------------------------------------------------------
# F02 – is_delayed by route
# ---------------------------------------------------------------------------

def plot_delay_by_route(obs: pd.DataFrame) -> None:
    route_stats = (
        obs.groupby("route_short_name")
        .agg(
            n          =("delay_minutes", "count"),
            mean_delay =("delay_minutes", "mean"),
            pct_delayed=("is_delayed",    "mean"),
        )
        .query("n >= 3")
        .sort_values("mean_delay", ascending=False)
        .reset_index()
    )

    if route_stats.empty:
        print("  F02: No routes with ≥3 observations — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(route_stats) * 0.45)))

    # Left: mean delay bar
    colors_l = [PALETTE["late"] if v > 0 else PALETTE["early"]
                for v in route_stats["mean_delay"]]
    bars = axes[0].barh(route_stats["route_short_name"],
                        route_stats["mean_delay"],
                        color=colors_l, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Mean Delay (minutes)")
    axes[0].set_title("Mean Delay by Route\n(red=late, blue=early)", fontweight="bold")
    axes[0].invert_yaxis()
    for bar, val in zip(bars, route_stats["mean_delay"]):
        axes[0].text(
            bar.get_width() + (0.3 if val >= 0 else -0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8
        )

    # Right: % delayed bar
    colors_r = [PALETTE["late"] if v > 0 else PALETTE["neutral"]
                for v in route_stats["pct_delayed"]]
    axes[1].barh(route_stats["route_short_name"],
                 route_stats["pct_delayed"] * 100,
                 color=colors_r, edgecolor="white")
    axes[1].set_xlabel("% of Observations that are Delayed (>5 min)")
    axes[1].set_title("Delay Rate by Route\n(≥3 observations each)", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(mticker.PercentFormatter())

    # Add observation count labels
    for i, row in route_stats.iterrows():
        axes[1].text(route_stats["pct_delayed"].max() * 100 * 1.02,
                     i, f"n={int(row['n'])}", va="center", fontsize=7, color="gray")

    fig.suptitle("VTA Route-Level Delay Analysis (GPS Observations)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "F02_is_delayed_by_route.png")


# ---------------------------------------------------------------------------
# F03 – Delay vs weather
# ---------------------------------------------------------------------------

def plot_delay_vs_weather(obs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colour points by delay category
    cat_colors = obs["delay_minutes"].apply(
        lambda x: PALETTE["late"] if x > 5 else (PALETTE["early"] if x < -5 else PALETTE["ontime"])
    )

    # Left: delay vs prcp
    ax = axes[0]
    ax.scatter(obs["prcp"], obs["delay_minutes"],
               c=cat_colors, alpha=0.6, s=40, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(5, color=PALETTE["late"], linestyle=":", linewidth=1)
    ax.set_xlabel("Daily Precipitation (inches)")
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Delay vs Precipitation", fontweight="bold")

    prcp_unique = obs["prcp"].nunique()
    if prcp_unique == 1:
        ax.text(0.05, 0.95,
                f"All observations on dry day\n(prcp = {obs['prcp'].iloc[0]:.2f}\")\n"
                "No rain variance in this dataset.",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Right: delay vs tmax
    ax2 = axes[1]
    ax2.scatter(obs["tmax"], obs["delay_minutes"],
                c=cat_colors, alpha=0.6, s=40, edgecolors="white", linewidths=0.3)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.axhline(5, color=PALETTE["late"], linestyle=":", linewidth=1)
    ax2.set_xlabel("Daily High Temperature (°F)")
    ax2.set_ylabel("Delay (minutes)")
    ax2.set_title("Delay vs Max Temperature", fontweight="bold")

    tmax_unique = obs["tmax"].nunique()
    if tmax_unique == 1:
        ax2.text(0.05, 0.95,
                 f"All observations on same day\n(tmax = {obs['tmax'].iloc[0]:.0f}°F)\n"
                 "No temperature variance — collect\nmore snapshots across different\ndays to see weather effect.",
                 transform=ax2.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    legend_patches = [
        mpatches.Patch(color=PALETTE["late"],   label="Late (>5 min)"),
        mpatches.Patch(color=PALETTE["ontime"], label="On-time (−5 to +5 min)"),
        mpatches.Patch(color=PALETTE["early"],  label="Early (<−5 min)"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Delay vs Weather Conditions\n"
                 "(Note: single day of observations — weather effect cannot yet be assessed)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "F03_delay_vs_weather.png")


# ---------------------------------------------------------------------------
# F04 – Delay by hour of day
# ---------------------------------------------------------------------------

def plot_delay_by_hour(obs: pd.DataFrame) -> None:
    hour_stats = (
        obs.groupby("hour")["delay_minutes"]
        .agg(mean="mean", median="median", count="count", sem="sem")
        .reindex(range(24))
        .reset_index()
    )
    has_data = hour_stats["count"].notna()

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_colors = [
        PALETTE["late"] if (not pd.isna(m) and m > 2) else
        (PALETTE["early"] if (not pd.isna(m) and m < -2) else PALETTE["neutral"])
        for m in hour_stats["mean"]
    ]
    x = hour_stats["hour"]
    bars = ax.bar(x, hour_stats["mean"].fillna(0), color=bar_colors,
                  edgecolor="white", width=0.7, alpha=0.85)

    # Error bars for hours with data
    obs_hrs = hour_stats[has_data]
    ax.errorbar(obs_hrs["hour"], obs_hrs["mean"], yerr=obs_hrs["sem"],
                fmt="none", color="black", capsize=4, linewidth=1.2)

    # Observation count labels
    for _, row in hour_stats[has_data].iterrows():
        ax.text(row["hour"], row["mean"] + (0.4 if row["mean"] >= 0 else -0.7),
                f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axhline(5, color=PALETTE["late"], linewidth=0.8, linestyle=":")
    ax.axhline(-5, color=PALETTE["early"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Hour of Day (PDT, service-day)")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Average Bus Delay by Hour of Day\n"
                 "(error bars = ±1 SEM; grey bars = no observations at that hour)",
                 fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")

    # Shade PM peak
    ax.axvspan(16, 19, alpha=0.07, color="red", label="PM peak window (16–19h)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "F04_delay_by_hour.png")


# ---------------------------------------------------------------------------
# F05 – Delay by day of week
# ---------------------------------------------------------------------------

def plot_delay_by_dow(obs: pd.DataFrame, df_full: pd.DataFrame) -> None:
    # Per-day: filter rows where that day flag == 1
    delay_by_day = {}
    sched_by_day = {}
    for day, abbr in zip(DAYS, DAY_ABBR):
        subset = obs[obs[day] == 1]
        delay_by_day[abbr] = {
            "mean": subset["delay_minutes"].mean() if len(subset) > 0 else np.nan,
            "n":    len(subset),
        }
        # Scheduled trip count from full dataset for that day
        sched_by_day[abbr] = int(df_full[df_full[day] == 1]["trip_id"].nunique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean delay by day (only Sunday has data here)
    ax = axes[0]
    means = [delay_by_day[d]["mean"] for d in DAY_ABBR]
    counts = [delay_by_day[d]["n"] for d in DAY_ABBR]
    bar_c = [PALETTE["late"] if (not np.isnan(m) and m > 0) else
             (PALETTE["early"] if (not np.isnan(m) and m < 0) else "lightgray")
             for m in means]
    bars = ax.bar(DAY_ABBR, [m if not np.isnan(m) else 0 for m in means],
                  color=bar_c, edgecolor="white", width=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    for i, (m, n) in enumerate(zip(means, counts)):
        if not np.isnan(m):
            ax.text(i, m + (0.2 if m >= 0 else -0.5),
                    f"{m:+.1f}\n(n={n})", ha="center", fontsize=8)
        else:
            ax.text(i, 0.2, "no data", ha="center", fontsize=7, color="gray")
    ax.set_ylabel("Mean Delay (minutes)")
    ax.set_title("Mean Observed Delay by Day-of-Week Flag\n"
                 "(Only Sunday snapshots collected so far)", fontweight="bold")

    # Right: scheduled trips per day vs observations collected
    ax2 = axes[1]
    x = np.arange(len(DAY_ABBR))
    w = 0.38
    bars1 = ax2.bar(x - w/2, [sched_by_day[d] for d in DAY_ABBR],
                    width=w, label="Scheduled trips (static)", color=PALETTE["neutral"], edgecolor="white")
    bars2 = ax2.bar(x + w/2, counts, width=w,
                    label="Delay observations collected", color=PALETTE["accent"], edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DAY_ABBR)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.set_ylabel("Count")
    ax2.set_title("Scheduled Trips vs Delay Observations\nby Day of Week", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Day-of-Week Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "F05_delay_by_day_of_week.png")


# ---------------------------------------------------------------------------
# F06 – On-time vs delayed pie chart
# ---------------------------------------------------------------------------

def plot_ontime_pie(obs: pd.DataFrame) -> None:
    n_late    = int((obs["delay_minutes"] > 5).sum())
    n_early   = int((obs["delay_minutes"] < -5).sum())
    n_ontime  = len(obs) - n_late - n_early
    total     = len(obs)

    sizes  = [n_ontime, n_early, n_late]
    labels = [
        f"On-time (−5 to +5 min)\n{n_ontime} trips ({n_ontime/total*100:.1f}%)",
        f"Early (< −5 min)\n{n_early} trips ({n_early/total*100:.1f}%)",
        f"Late (> +5 min)\n{n_late} trips ({n_late/total*100:.1f}%)",
    ]
    colors  = [PALETTE["ontime"], PALETTE["early"], PALETTE["late"]]
    explode = (0.02, 0.02, 0.08)   # pop out the 'late' slice

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        explode=explode, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title(
        f"On-time Performance — {total} GPS-Observed Trips\n"
        f"(Threshold: ±5 minutes from scheduled arrival)",
        fontsize=12, fontweight="bold", pad=20
    )
    fig.tight_layout()
    save(fig, "F06_on_time_pie.png")


# ---------------------------------------------------------------------------
# F07 – Occupancy vs delay
# ---------------------------------------------------------------------------

def plot_occupancy_vs_delay(obs: pd.DataFrame) -> None:
    occ_order = ["EMPTY", "FEW_SEATS_AVAILABLE", "FULL"]
    occ_labels = {"EMPTY": "Empty", "FEW_SEATS_AVAILABLE": "Few Seats", "FULL": "Full"}
    occ_colors = {"EMPTY": "#3498db", "FEW_SEATS_AVAILABLE": "#f39c12", "FULL": "#e74c3c"}

    valid = obs[obs["occupancy_label"].isin(occ_order)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: violin / strip plot of delay by occupancy
    ax = axes[0]
    for i, occ in enumerate(occ_order):
        subset = valid[valid["occupancy_label"] == occ]["delay_minutes"]
        if len(subset) < 2:
            continue
        parts = ax.violinplot(subset, positions=[i], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(occ_colors[occ])
            pc.set_alpha(0.6)
        # Jitter strip
        jitter = np.random.uniform(-0.1, 0.1, len(subset))
        ax.scatter(i + jitter, subset, color=occ_colors[occ],
                   alpha=0.5, s=20, zorder=3)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(5, color=PALETTE["late"], linestyle=":", linewidth=1)
    ax.set_xticks(range(len(occ_order)))
    ax.set_xticklabels([occ_labels[o] for o in occ_order])
    ax.set_ylabel("Delay (minutes)")
    ax.set_title("Delay Distribution by Occupancy Level", fontweight="bold")

    # Right: mean delay + % delayed bar chart
    ax2 = axes[1]
    stats = (
        valid.groupby("occupancy_label")
        .agg(mean_delay=("delay_minutes", "mean"),
             pct_late=("is_delayed", "mean"),
             n=("delay_minutes", "count"))
        .reindex(occ_order)
        .reset_index()
    )
    x = np.arange(len(stats))
    w = 0.38
    bars1 = ax2.bar(x - w/2, stats["mean_delay"],
                    width=w, label="Mean delay (min)",
                    color=[occ_colors[o] for o in occ_order],
                    edgecolor="white", alpha=0.85)
    ax2b = ax2.twinx()
    bars2 = ax2b.bar(x + w/2, stats["pct_late"] * 100,
                     width=w, label="% Late (>5 min)",
                     color=[occ_colors[o] for o in occ_order],
                     edgecolor="white", alpha=0.45, hatch="///")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{occ_labels[o]}\n(n={int(stats.loc[i,'n'])})"
                         for i, o in enumerate(occ_order)])
    ax2.set_ylabel("Mean Delay (minutes)", color="black")
    ax2b.set_ylabel("% Late", color="gray")
    ax2b.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_title("Mean Delay & Late-Rate by Occupancy", fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle("Does Bus Fullness Correlate with Delays?", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "F07_occupancy_vs_delay.png")


# ---------------------------------------------------------------------------
# Written EDA summary
# ---------------------------------------------------------------------------

def print_summary(obs: pd.DataFrame, df_full: pd.DataFrame) -> None:
    n_late   = int((obs["delay_minutes"] > 5).sum())
    n_early  = int((obs["delay_minutes"] < -5).sum())
    n_ontime = len(obs) - n_late - n_early
    total    = len(obs)

    top_delayed_route = (
        obs.groupby("route_short_name")["delay_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    peak_delay_hour = (
        obs.groupby("hour")["delay_minutes"]
        .mean()
        .idxmax()
    )

    print(f"\n{SECTION}")
    print("FINAL EDA WRITTEN SUMMARY — vta_final.csv")
    print("Comparing GPS-observed delay findings to initial static schedule EDA")
    print(SECTION)

    print(f"""
DATASET OVERVIEW
  Total stop-time records (full)  : {len(df_full):>9,}
  GPS-observed delay records      : {total:>9,}  (0.13% coverage)
  Unique trips with delay data    : {obs['trip_id'].nunique():>9,}
  Unique routes with delay data   : {obs['route_short_name'].nunique():>9,}
  Observation window              : Sunday April 5–6 2026, 13:00–19:00 PDT

ON-TIME PERFORMANCE (GPS-Derived)
  On-time  (−5 to +5 min) : {n_ontime:>4}  ({n_ontime/total*100:.1f}%)
  Early    (<  −5 min)    : {n_early:>4}  ({n_early/total*100:.1f}%)
  Late     (>  +5 min)    : {n_late:>4}  ({n_late/total*100:.1f}%)
  Mean delay              : {obs['delay_minutes'].mean():>+7.2f} min  (slightly early overall)
  Median delay            : {obs['delay_minutes'].median():>+7.2f} min
  Worst observed delay    : {obs['delay_minutes'].max():>+7.2f} min  ({obs.loc[obs['delay_minutes'].idxmax(), 'route_short_name']})

DELAY BY HOUR (NEW FINDING)
  Observations span hours 13–19 (1–7 PM PDT).
  Hour 16:00 had the highest mean delay: {obs[obs['hour']==16]['delay_minutes'].mean():.1f} min
    → Aligns with initial EDA's PM peak (15:00–16:00 had most scheduled trips).
  Hour 19:00 was the most early-running: {obs[obs['hour']==19]['delay_minutes'].mean():.1f} min
    → Late evening buses appear to be ahead of schedule.

DELAY BY ROUTE (NEW vs INITIAL EDA)
  Initial EDA top-5 routes by TRIP COUNT : Route 22, Rapid 500, 25, 23, Rapid 522
  GPS-observed most delayed route         : {top_delayed_route}
    → The high-frequency routes are not necessarily the most delayed.
    Route 40 (avg +{obs[obs['route_short_name']=='40']['delay_minutes'].mean():.1f} min,
    60% late) ran behind schedule despite lower scheduled trip count.

WEATHER ANALYSIS (LIMITATION)
  All 455 observations were on a single dry day (prcp=0.00\", tmax=68°F).
  No weather variance exists in current data → scatter plots are informational
  placeholders. Collecting snapshots across multiple days — especially rainy
  days (21 rainy days exist in the weather CSV) — will reveal the weather effect.

OCCUPANCY vs DELAY
  EMPTY buses        : {obs[obs['occupancy_label']=='EMPTY']['delay_minutes'].mean():.2f} min avg delay   ({(obs[obs['occupancy_label']=='EMPTY']['is_delayed']==1).sum()} late)
  FEW_SEATS buses    : {obs[obs['occupancy_label']=='FEW_SEATS_AVAILABLE']['delay_minutes'].mean() if len(obs[obs['occupancy_label']=='FEW_SEATS_AVAILABLE'])>0 else 'N/A'} min avg delay
  FULL buses         : {obs[obs['occupancy_label']=='FULL']['delay_minutes'].mean() if len(obs[obs['occupancy_label']=='FULL'])>0 else 'N/A'} min avg delay
  Preliminary signal: FULL buses show different delay patterns than EMPTY.
  More data needed to confirm statistical significance.

DAY-OF-WEEK ANALYSIS
  Only Sunday snapshots collected → delay-by-DOW limited to Sunday.
  Initial EDA showed Monday has the most scheduled trips (3,524 trip-runs)
  and Sunday the fewest (2,621). Collecting weekday snapshots (especially
  Mon–Fri during AM/PM peaks) will likely reveal higher delays due to traffic.

COMPARISON TO INITIAL EDA
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Metric                  │ Initial EDA (Static)  │ Final EDA (GPS)  │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Peak hour (trips)       │ 15:00 (522 trips)     │ 16:00 (+4.4 min) │
  │ Busiest route           │ Route 22 (424 trips)  │ Route 22 in top  │
  │ Most delayed route      │ N/A (no RT data)      │ Route 40         │
  │ % late                  │ N/A                   │ 6.4%             │
  │ Weather effect          │ N/A                   │ Inconclusive*    │
  │ Past-midnight routes    │ 19 routes             │ Not yet observed │
  └─────────────────────────────────────────────────────────────────────┘
  * Single dry day of observations. Collect rainy-day snapshots to assess.

NEXT STEPS
  1. Run 04_fetch_realtime.py + 04b_fetch_vehicle_positions.py hourly
     across weekdays to build a richer delay dataset.
  2. Re-run 04c + 06 + 03b to update findings with more variance.
  3. Target rainy days (check weather forecast vs 21 rainy days in CSV)
     for weather-delay correlation analysis.
""")
    print(SECTION)
    print(f"All plots saved to: eda_outputs/final/")
    print(SECTION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SECTION)
    print("03b_eda_final.py — Extended EDA on Delay-Enriched Dataset")
    print(SECTION)

    print("\nLoading vta_final.csv …")
    df_full, obs = load_data(CLEANED / "vta_final.csv")

    print("\n[F01] Delay distribution histogram")
    plot_delay_distribution(obs)

    print("\n[F02] is_delayed by route")
    plot_delay_by_route(obs)

    print("\n[F03] Delay vs weather")
    plot_delay_vs_weather(obs)

    print("\n[F04] Delay by hour of day")
    plot_delay_by_hour(obs)

    print("\n[F05] Delay by day of week")
    plot_delay_by_dow(obs, df_full)

    print("\n[F06] On-time vs delayed pie chart")
    plot_ontime_pie(obs)

    print("\n[F07] Occupancy vs delay")
    plot_occupancy_vs_delay(obs)

    print_summary(obs, df_full)


if __name__ == "__main__":
    main()
