# VTA Bus Delay Prediction — DATA 245 Group Project

---

## 1. Project Overview

This project builds a machine learning system to predict whether a VTA (Santa Clara Valley Transportation Authority) bus will be more than 5 minutes late at a given stop. Late buses cause missed connections, reduce rider trust, and make public transit feel unreliable — but delays are not random. They correlate with the time of day, the day of week, weather conditions, how crowded the bus is, and which route is being served. By combining the official VTA schedule (GTFS static), real-time GPS vehicle positions, and historical weather data for San Jose, we create a labeled dataset where each row represents one bus at one stop, enriched with features that a machine learning model can learn from. The end goal is a classifier that transit planners or a real-time app could use to warn riders about likely delays before they happen.

---

## 2. Data Sources

| Source | What It Is | How We Got It |
|---|---|---|
| **VTA GTFS Static** | Official scheduled timetables, routes, stops, and calendar for all VTA buses | Downloaded from the [511 SF Bay API](https://511.org/open-data/transit) — agency code `SC` |
| **NOAA Daily Weather** | Daily precipitation (inches), high/low temperature (°F) for San Jose, CA | Downloaded from [NOAA Climate Data Online](https://www.ncdc.noaa.gov/cdo-web/) — station USC00047821, file `4277464.csv` |
| **GTFS-RT Vehicle Positions** | Live GPS coordinates, speed, occupancy, and stop position for every active VTA bus | Fetched in real time from `https://api.511.org/transit/vehiclepositions?agency=SC` using the `gtfs-realtime-bindings` protobuf library |

---

## 3. Folder Structure

```
GROUP_PROJECT/
│
├── 01_data_loading.py            Load + merge GTFS static files, filter to VTA bus routes
├── 02_data_cleaning.py           Clean nulls, fix times, join calendar + weather flags
├── 03_eda.py                     Initial EDA on the static schedule (7 plots)
├── 03b_eda_final.py              Final EDA on the delay-enriched dataset (7 plots)
├── 04_fetch_realtime.py          Fetch GTFS-RT TripUpdates snapshot → realtime_snapshot_N.csv
├── 04b_fetch_vehicle_positions.py  Fetch live GPS vehicle positions → vehicle_positions_N.csv
├── 04c_calculate_delays.py       Calculate actual delays from GPS vs schedule, all snapshots
├── 05_merge_weather.py           Join NOAA weather onto the GTFS DataFrame
├── 06_merge_realtime.py          Merge calculated delays into the final dataset
│
├── GTFSTransitData_SC/           Original GTFS static .txt files (raw download, untouched)
│
├── data/
│   ├── raw/
│   │   ├── stop_times.txt                 GTFS: every scheduled stop for every trip
│   │   ├── trips.txt                      GTFS: trip → route mapping + service_id
│   │   ├── routes.txt                     GTFS: route names, types, URLs
│   │   ├── stops.txt                      GTFS: stop names, lat/lon
│   │   ├── calendar.txt                   GTFS: which days each service_id runs
│   │   ├── calendar_dates.txt             GTFS: holiday/exception overrides
│   │   ├── 4277464.csv                    NOAA weather: daily precip + temp, Oct 2025–Apr 2026
│   │   ├── realtime_snapshot_1.csv        GTFS-RT TripUpdates snapshot 1 (pre-dawn)
│   │   ├── realtime_snapshot_2.csv        GTFS-RT TripUpdates snapshot 2 (afternoon)
│   │   ├── realtime_snapshot_3.csv        GTFS-RT TripUpdates snapshot 3
│   │   ├── vehicle_positions_1.csv        GPS vehicle positions snapshot 1
│   │   ├── vehicle_positions_2.csv        GPS vehicle positions snapshot 2
│   │   ├── vehicle_positions_3.csv        GPS vehicle positions snapshot 3
│   │   ├── calculated_delays_1.csv        Delays from snapshot 1 only (vehicle_positions_1)
│   │   └── calculated_delays_all.csv      Delays stacked across all snapshots, deduped
│   │
│   └── cleaned/
│       ├── stop_times_merged.parquet      stop_times + trips + routes merged (VTA bus only)
│       ├── vta_cleaned.parquet            Cleaned + calendar + exception flags (no weather)
│       ├── vta_with_weather.parquet       vta_cleaned + NOAA weather joined on start_date
│       ├── vta_final.parquet              Final model-ready dataset (with delay columns)
│       └── vta_final.csv                 Same as vta_final.parquet but as CSV for Nikhil
│
├── eda_outputs/
│   ├── 01_trips_by_hour.png              Trips scheduled per hour of day
│   ├── 02_trips_by_day_of_week.png       Trips per day of week (weekday vs weekend)
│   ├── 03_route_analysis.png             Top routes by stop count and trip count
│   ├── 04_busiest_stops.png              Top 20 busiest stops by trip throughput
│   ├── 05_calendar_heatmap.png           Route × day-of-week heatmap
│   ├── 06_past_midnight_routes.png       Routes with past-midnight trips
│   ├── 07_avg_trip_length.png            Average stops per trip by route
│   └── final/
│       ├── F01_delay_distribution.png    Histogram of actual delay_minutes
│       ├── F02_is_delayed_by_route.png   Delay rate and mean delay per route
│       ├── F03_delay_vs_weather.png      Scatter: delay vs precipitation and temperature
│       ├── F04_delay_by_hour.png         Average delay by hour of day
│       ├── F05_delay_by_day_of_week.png  Delay by day-of-week flag + scheduled trips
│       ├── F06_on_time_pie.png           On-time / early / late pie chart
│       └── F07_occupancy_vs_delay.png    Delay by bus occupancy level
│
└── README.md                             This file
```

---

## 4. What Has Been Done — Naman

Each script picks up where the last one left off. Run them in order.

- **`01_data_loading.py`**
  Reads the six raw GTFS `.txt` files from `data/raw/`. Filters out School trippers and the Caltrain CTBUS route so we only keep the 57 regular VTA bus routes. Merges `stop_times` → `trips` → `routes` into one flat table with 361,885 rows (one row per bus stop visit). Saves to `data/cleaned/stop_times_merged.parquet`.

- **`02_data_cleaning.py`**
  Loads the merged parquet. Prints a full missing-value audit. Fixes GTFS times that go past midnight (e.g. `25:30:00` → 91,800 seconds). Joins day-of-week service flags (monday–sunday) and service date range from `calendar.txt`. Adds exception count flags from `calendar_dates.txt` (holidays, added/removed service). Drops 10 columns that are useless for ML (all-null, constant, cosmetic). Saves to `data/cleaned/vta_cleaned.parquet`.

- **`03_eda.py`**
  Loads `vta_cleaned.parquet` and produces 7 static-schedule plots saved to `eda_outputs/`. Key findings: peak scheduling hour is 15:00, busiest route is Route 22, busiest stops are at Santa Clara & 5th Street (downtown SJ), 19 routes run past midnight.

- **`04_fetch_realtime.py`**
  Calls the 511.org GTFS-RT TripUpdates API and saves a snapshot CSV. Each run auto-increments the filename (`realtime_snapshot_1.csv`, `_2.csv`, …). Note: this feed returns delay = 0 for all stops because VTA publishes schedule-based times, not GPS-derived delays. Use `04b` and `04c` for real delay data.

- **`04b_fetch_vehicle_positions.py`**
  Calls the 511.org GTFS-RT VehiclePositions API and saves real GPS-tracked vehicles only (filters out `schedBased` vehicle IDs). Each run auto-increments to `vehicle_positions_N.csv`. Captures: vehicle ID, trip, route, direction, GPS lat/lon, speed, bearing, occupancy, and current stop position. This is the source of truth for actual delays.

- **`04c_calculate_delays.py`**
  Loops through **all** `vehicle_positions_*.csv` files. For each one, joins to the GTFS schedule on `(trip_id, stop_sequence)` and calculates: `actual_delay_seconds = GPS_timestamp_local − scheduled_arrival`. Stacks all snapshots together, then deduplicates on `(trip_id, stop_sequence)` keeping the row with the largest absolute delay per stop. Saves to `data/raw/calculated_delays_all.csv`.

- **`05_merge_weather.py`**
  Loads `vta_cleaned.parquet` and `data/raw/4277464.csv`. Joins weather data onto the GTFS table using `start_date` as the key (the first date of each service window). Adds `prcp`, `tmax`, `tmin`, `is_rainy` (1 if rain > 0.1"), and `temp_range`. Saves to `data/cleaned/vta_with_weather.parquet`.

- **`06_merge_realtime.py`**
  Loads `calculated_delays_all.csv` and `vta_with_weather.parquet`. Joins on `(trip_id, stop_sequence)`. Brings in `delay_minutes`, `is_delayed`, `speed_mph`, and `occupancy_label`. Saves the final model-ready dataset to both `data/cleaned/vta_final.parquet` and `data/cleaned/vta_final.csv`.

- **`03b_eda_final.py`**
  Loads `vta_final.csv` and produces 7 final EDA plots on the delay-enriched data. Prints a written summary comparing GPS-observed findings to the initial static-schedule EDA.

---

## 5. Final Dataset — `vta_final.csv`

**361,885 rows × 45 columns**

Each row represents one bus stopping at one stop on one scheduled trip. Most rows have `NaN` for the delay columns because we have only collected a handful of GPS snapshots so far — more snapshots will fill in more rows over time.

### Key columns explained

| Column | Type | Description |
|---|---|---|
| `trip_id` | string | Unique identifier for one scheduled bus run (matches GTFS) |
| `route_id` / `route_short_name` | string | Route identifier and human-readable name (e.g. `22`, `Rapid 500`) |
| `stop_id` | string | Which physical stop |
| `stop_sequence` | int | Position of this stop within the trip (1 = first stop) |
| `arrival_sec` | int | Scheduled arrival time in seconds since service-day midnight |
| `direction_id` | int | 0 = outbound, 1 = inbound |
| `monday` … `sunday` | Int8 | 1 if this trip's service runs on that day, 0 otherwise |
| `start_date` / `end_date` | datetime | The date window for this service schedule |
| `has_exception` | int | 1 if this service_id has any holiday/exception overrides |
| `prcp` | float | Daily precipitation in inches on `start_date` |
| `tmax` / `tmin` | float | Daily high / low temperature °F |
| `is_rainy` | Int8 | 1 if `prcp > 0.1"` (meaningful rain), else 0 |
| `temp_range` | Int16 | `tmax − tmin` — a proxy for weather variability |
| `delay_minutes` | float | **GPS-derived actual delay** in minutes (+= late, −= early). NaN if not observed yet |
| `is_delayed` | Int8 | **Target variable** — 1 if `delay_minutes > 5`, else 0 |
| `speed_mph` | float | Vehicle speed at time of GPS observation |
| `occupancy_label` | string | Passenger load: EMPTY / FEW_SEATS_AVAILABLE / FULL |
| `current_status_name` | string | INCOMING_AT / IN_TRANSIT_TO / STOPPED_AT |
| `delay_snapshot_ts` | string | UTC timestamp when this GPS reading was captured |

---

## 6. Target Variable — `is_delayed`

**`is_delayed = 1`** means the bus was observed more than **5 minutes late** compared to its scheduled arrival time at that stop.  
**`is_delayed = 0`** means on-time, early, or not yet observed.

The 5-minute threshold matches the industry standard used by transit agencies (GTFS-RT spec, FTA performance benchmarks).

### Current class distribution (455 observed trips)

| Class | Count | % |
|---|---|---|
| On-time / early (`is_delayed = 0`) | 426 | 93.6% |
| Late (`is_delayed = 1`) | 29 | 6.4% |

**This is imbalanced.** A naive model that always predicts "on-time" would get 93.6% accuracy but be useless. Nikhil needs to apply SMOTE (see below) to fix this before training.

---

## 7. Nikhil's Tasks — Feature Engineering

**What feature engineering means:** The raw data has columns like `arrival_sec` (a number like 49740) or `monday = 1`. Feature engineering means turning these raw values into new, smarter columns that help the ML model understand patterns — for example, converting `arrival_sec` into a "is this the rush hour?" flag.

### Load the data

```python
import pandas as pd
df = pd.read_csv("data/cleaned/vta_final.csv", low_memory=False)

# Only use rows where we have real delay observations for training
obs = df[df["delay_minutes"].notna()].copy()
```

### Features to create

1. **`is_rush_hour`** — Set to 1 if the scheduled arrival is during AM peak (6–9 AM, i.e. `arrival_sec` between 21600–32400) or PM peak (4–7 PM, i.e. 57600–68400).
   ```python
   obs["hour"] = obs["arrival_sec"] // 3600
   obs["is_rush_hour"] = obs["hour"].between(6, 9) | obs["hour"].between(16, 19)
   obs["is_rush_hour"] = obs["is_rush_hour"].astype(int)
   ```

2. **`is_weekend`** — 1 if `saturday == 1` or `sunday == 1`, else 0.
   ```python
   obs["is_weekend"] = ((obs["saturday"] == 1) | (obs["sunday"] == 1)).astype(int)
   ```

3. **`hour_of_day`** — Just the hour (0–23) as a numeric feature.
   ```python
   obs["hour_of_day"] = obs["arrival_sec"] // 3600
   ```

4. **`headway_deviation`** (advanced) — For each route, compute the average time between consecutive trips at the same stop, then flag stops where the gap is unusually large or small. A large gap means the next bus is far away and this bus might be absorbing extra passengers.

5. **`route_stop_count`** — How many unique stops does this route serve? Routes with more stops tend to accumulate delay. Join from the static schedule:
   ```python
   stop_counts = df.groupby("route_short_name")["stop_id"].nunique().rename("route_stop_count")
   obs = obs.join(stop_counts, on="route_short_name")
   ```

6. **`stop_position_pct`** — `stop_sequence / max_stop_sequence` for that trip. Buses tend to get later the further into a trip you go.
   ```python
   trip_len = df.groupby("trip_id")["stop_sequence"].max().rename("trip_length")
   obs = obs.join(trip_len, on="trip_id")
   obs["stop_position_pct"] = obs["stop_sequence"] / obs["trip_length"]
   ```

### Handle class imbalance with SMOTE

```python
pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

feature_cols = ["hour_of_day", "is_rush_hour", "is_weekend", "route_stop_count",
                "stop_position_pct", "tmax", "prcp", "is_rainy", "temp_range",
                "stop_sequence", "direction_id"]

X = obs[feature_cols].fillna(0)
y = obs["is_delayed"]

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
```

SMOTE creates synthetic "delayed" examples so the model sees a balanced 50/50 split instead of 94/6.

---

## 8. Naeem's Tasks — Modeling

**What the modeling step means:** We train several ML models on Nikhil's engineered features to predict `is_delayed`. We then compare them to find the best one and explain which features matter most.

### Train / test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
```

### Models to train

| Model | Why |
|---|---|
| **Logistic Regression** | Simple baseline — easy to interpret, fast to train |
| **Random Forest** | Handles non-linear patterns, gives feature importances for free |
| **SVM (RBF kernel)** | Works well on small datasets with class imbalance after SMOTE |
| *(Optional) XGBoost* | Often the best performer on tabular data |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
}
```

### Evaluation metrics

Do **not** use accuracy — it is misleading with imbalanced classes. Use:

| Metric | What it tells you |
|---|---|
| **F1 Score** | Balance between catching late buses and not crying wolf |
| **ROC-AUC** | How well the model separates delayed from on-time overall |
| **Precision** | Of the buses we predicted as late, how many actually were? |
| **Recall** | Of all the buses that were late, how many did we catch? |

```python
from sklearn.metrics import classification_report, roc_auc_score
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

### K-Means clustering on stops

Group the 3,032 stops into clusters based on delay behaviour, location, and route density. This helps identify "delay hotspot" areas of the network.

```python
from sklearn.cluster import KMeans

# Join stop lat/lon from stops.txt, then cluster on:
# [latitude, longitude, avg_delay_at_stop, trips_per_day]
kmeans = KMeans(n_clusters=8, random_state=42)
stop_features["cluster"] = kmeans.fit_predict(stop_features)
```

### SHAP feature importance

After training Random Forest (or XGBoost), use SHAP to explain which features drive delay predictions the most.

```python
pip install shap

import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)
```

---

## 9. How to Run Everything

Run these steps in order from inside the `GROUP_PROJECT/` folder.

```bash
# 0. Install dependencies
pip install pandas pyarrow gtfs-realtime-bindings requests matplotlib seaborn imbalanced-learn shap

# 1. Load and merge the GTFS static files
python3 01_data_loading.py
# Output: data/cleaned/stop_times_merged.parquet

# 2. Clean the merged data
python3 02_data_cleaning.py
# Output: data/cleaned/vta_cleaned.parquet

# 3. Initial EDA (static schedule)
python3 03_eda.py
# Output: eda_outputs/*.png

# 4. Join NOAA weather
python3 05_merge_weather.py
# Output: data/cleaned/vta_with_weather.parquet

# 5. Collect GPS snapshots (run multiple times, ideally during service hours 6AM–9PM)
python3 04b_fetch_vehicle_positions.py
# Output: data/raw/vehicle_positions_N.csv  (auto-increments each run)

# 6. Calculate delays from all GPS snapshots
python3 04c_calculate_delays.py
# Output: data/raw/calculated_delays_all.csv

# 7. Build the final dataset
python3 06_merge_realtime.py
# Output: data/cleaned/vta_final.parquet  +  data/cleaned/vta_final.csv

# 8. Final EDA on the delay-enriched dataset
python3 03b_eda_final.py
# Output: eda_outputs/final/*.png
```

> **Re-running after collecting more snapshots:** Just run steps 5 → 6 → 7 → 8 again. Step 6 automatically stacks all `vehicle_positions_*.csv` files and step 7 rebuilds `vta_final.*` from scratch.

---

## 10. Important Notes

- **GPS coverage is currently 0.13%** — only 455 of 361,885 stop-time rows have a real delay observation. This is because we have collected only a handful of GPS snapshots so far. More snapshots = more labelled rows = better model.

- **All snapshots are from Sunday** (April 5–6, 2026). To get weekday and peak-hour delay patterns, run `04b_fetch_vehicle_positions.py` during **Monday–Friday, 7–9 AM and 4–7 PM**.

- **Snapshots auto-stack** — every time you run `04b`, it saves a new numbered file. Every time you run `04c`, it reads all `vehicle_positions_*.csv` files, combines them, deduplicates, and saves `calculated_delays_all.csv`. You never need to merge manually.

- **The TripUpdates feed (`04_fetch_realtime.py`) always returns delay = 0** for VTA because the agency publishes schedule-based updates, not GPS-derived delays. Use `04b` + `04c` for real delay data.

- **Weather variance is low right now** — all observations so far are from a single dry day (68°F, 0" rain). The NOAA CSV contains 21 rainy days between Oct 2025–Apr 2026. Collecting snapshots on a rainy day will unlock the weather feature.

- **File sizes** — `vta_final.csv` is large (~150 MB). Use the `.parquet` version for faster loading in Python:
  ```python
  df = pd.read_parquet("data/cleaned/vta_final.parquet")
  ```

---

*Last updated: April 2026 — Naman Vipul Chheda*
