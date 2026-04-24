from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import FINAL_DATA_CSV

NUMERIC_FEATURES = [
    'hour_of_day',
    'is_rush_hour',
    'is_weekend',
    'route_stop_count',
    'trip_length',
    'stop_position_pct',
    'stop_sequence',
    'direction_id',
    'speed_mph',
]

CATEGORICAL_FEATURES = [
    'route_short_name',
    'occupancy_label',
    'current_status_name',
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_final_data(csv_path: Path = FINAL_DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['trip_id'] = df['trip_id'].astype(str)
    df['stop_id'] = df['stop_id'].astype(str)
    df['hour_of_day'] = (pd.to_numeric(df['arrival_sec'], errors='coerce') // 3600).astype('Int64')
    df['is_rush_hour'] = (
        df['hour_of_day'].between(6, 9) | df['hour_of_day'].between(16, 19)
    ).astype('Int64')
    df['is_weekend'] = (
        (pd.to_numeric(df['saturday'], errors='coerce').fillna(0) == 1)
        | (pd.to_numeric(df['sunday'], errors='coerce').fillna(0) == 1)
    ).astype('Int64')

    route_stop_count = (
        df.groupby('route_short_name', observed=False)['stop_id']
        .nunique()
        .rename('route_stop_count')
    )
    trip_length = (
        df.groupby('trip_id', observed=False)['stop_sequence']
        .max()
        .rename('trip_length')
    )

    df = df.join(route_stop_count, on='route_short_name')
    df = df.join(trip_length, on='trip_id')

    df['stop_sequence'] = pd.to_numeric(df['stop_sequence'], errors='coerce')
    df['direction_id'] = pd.to_numeric(df['direction_id'], errors='coerce')
    df['speed_mph'] = pd.to_numeric(df['speed_mph'], errors='coerce')
    df['route_stop_count'] = pd.to_numeric(df['route_stop_count'], errors='coerce')
    df['trip_length'] = pd.to_numeric(df['trip_length'], errors='coerce')
    df['stop_position_pct'] = df['stop_sequence'] / df['trip_length']

    if 'delay_snapshot_ts' in df.columns:
        snapshot = pd.to_datetime(df['delay_snapshot_ts'], utc=True, errors='coerce')
        df['observation_timestamp_local'] = snapshot.dt.tz_convert('America/Los_Angeles')
        df['observation_date_local'] = df['observation_timestamp_local'].dt.date.astype('string')
    else:
        df['observation_timestamp_local'] = pd.NaT
        df['observation_date_local'] = pd.Series(pd.NA, index=df.index, dtype='string')

    df['is_delayed'] = pd.to_numeric(df['is_delayed'], errors='coerce').fillna(0).astype(int)
    df['delay_minutes'] = pd.to_numeric(df['delay_minutes'], errors='coerce')

    return df


def get_observed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[df['delay_minutes'].notna()].copy()
    obs['occupancy_label'] = obs['occupancy_label'].fillna('UNKNOWN')
    obs['current_status_name'] = obs['current_status_name'].fillna('UNKNOWN')
    obs['route_short_name'] = obs['route_short_name'].fillna('UNKNOWN')
    return obs


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                'cat',
                Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_modeling_frame() -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_df = engineer_features(load_final_data())
    observed = get_observed_dataset(full_df)
    return full_df, observed
