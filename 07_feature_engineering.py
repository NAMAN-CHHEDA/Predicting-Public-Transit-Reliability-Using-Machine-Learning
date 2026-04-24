import pandas as pd
from imblearn.over_sampling import SMOTE

df = pd.read_parquet("data/cleaned/vta_final.parquet")

# Keep only rows with observed delay labels
obs = df[df["delay_minutes"].notna()].copy()

# Time-based features
obs["hour_of_day"] = obs["arrival_sec"] // 3600
obs["is_rush_hour"] = (
    obs["hour_of_day"].between(6, 9) | obs["hour_of_day"].between(16, 19)
).astype(int)

obs["is_weekend"] = (
    (obs["saturday"] == 1) | (obs["sunday"] == 1)
).astype(int)

# Route-level feature
stop_counts = (
    df.groupby("route_short_name", observed=False)["stop_id"]
      .nunique()
      .rename("route_stop_count")
)
obs = obs.join(stop_counts, on="route_short_name")

# Trip progress feature
trip_len = (
    df.groupby("trip_id", observed=False)["stop_sequence"]
      .max()
      .rename("trip_length")
)
obs = obs.join(trip_len, on="trip_id")
obs["stop_position_pct"] = obs["stop_sequence"] / obs["trip_length"]

feature_cols = [
    "hour_of_day",
    "is_rush_hour",
    "is_weekend",
    "route_stop_count",
    "stop_position_pct",
    "tmax",
    "prcp",
    "is_rainy",
    "temp_range",
    "stop_sequence",
    "direction_id",
    "speed_mph",
]

# Force plain numeric dtypes before SMOTE
X = obs[feature_cols].copy()
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")

y = pd.to_numeric(obs["is_delayed"], errors="coerce").fillna(0).astype("int64")

print("X dtypes before SMOTE:")
print(X.dtypes)
print("\nClass counts before SMOTE:")
print(y.value_counts())

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X.to_numpy(), y.to_numpy())

X_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
y_resampled = pd.Series(y_resampled, name="is_delayed")

engineered = X_resampled.copy()
engineered["is_delayed"] = y_resampled

engineered.to_parquet("data/cleaned/vta_features.parquet", index=False)
engineered.to_csv("data/cleaned/vta_features.csv", index=False)

print("\nSaved engineered datasets:")
print("data/cleaned/vta_features.parquet")
print("data/cleaned/vta_features.csv")
print("\nClass counts after SMOTE:")
print(engineered["is_delayed"].value_counts())
print(engineered.head())