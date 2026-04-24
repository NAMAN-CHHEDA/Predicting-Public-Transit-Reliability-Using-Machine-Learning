from __future__ import annotations

import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import load_modeling_frame
from config import OUTPUTS_DIR, STOPS_CSV


SEED = 42


def main() -> None:
    _, observed = load_modeling_frame()
    observed['late_minutes_only'] = observed['delay_minutes'].clip(lower=0)

    stop_table = pd.read_csv(STOPS_CSV, low_memory=False)
    stop_table['stop_id'] = stop_table['stop_id'].astype(str)

    stop_features = (
        observed.groupby('stop_id', observed=False)
        .agg(
            mean_late_minutes=('late_minutes_only', 'mean'),
            max_late_minutes=('late_minutes_only', 'max'),
            late_rate=('is_delayed', 'mean'),
            observations=('delay_minutes', 'count'),
            mean_speed=('speed_mph', 'mean'),
        )
        .reset_index()
    )

    stop_features = stop_features[stop_features['observations'] >= 2].copy()

    clustering_features = ['mean_late_minutes', 'max_late_minutes', 'late_rate', 'observations']
    X_scaled = StandardScaler().fit_transform(stop_features[clustering_features])

    model = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    stop_features['cluster_id'] = model.fit_predict(X_scaled)

    cluster_summary = (
        stop_features.groupby('cluster_id')[clustering_features]
        .mean()
        .sort_values(['mean_late_minutes', 'late_rate', 'max_late_minutes'], ascending=False)
        .reset_index()
    )
    cluster_summary['severity_rank'] = range(1, len(cluster_summary) + 1)

    stop_features = stop_features.merge(
        cluster_summary[['cluster_id', 'severity_rank']], on='cluster_id', how='left'
    )
    stop_features = stop_features.merge(
        stop_table[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id', how='left'
    )

    stop_features.sort_values(
        ['severity_rank', 'mean_late_minutes', 'late_rate', 'max_late_minutes'],
        ascending=[True, False, False, False],
        inplace=True,
    )

    stop_features.to_csv(OUTPUTS_DIR / 'stop_clusters.csv', index=False)
    cluster_summary.to_csv(OUTPUTS_DIR / 'cluster_summary.csv', index=False)
    stop_features[stop_features['severity_rank'] == 1].head(20).to_csv(
        OUTPUTS_DIR / 'top_bottleneck_stops.csv', index=False
    )

    with open(OUTPUTS_DIR / 'clustering_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'cluster_count': int(cluster_summary.shape[0]),
                'cluster_rows': int(stop_features.shape[0]),
                'top_cluster_average_late_minutes': float(cluster_summary.iloc[0]['mean_late_minutes']),
                'top_cluster_average_late_rate': float(cluster_summary.iloc[0]['late_rate']),
            },
            f,
            indent=2,
        )

    print(cluster_summary.to_string(index=False))
    print('Saved stop cluster outputs.')


if __name__ == '__main__':
    main()
