from __future__ import annotations

import json

from common import FEATURE_COLUMNS, load_modeling_frame
from config import OUTPUTS_DIR


def main() -> None:
    full_df, observed = load_modeling_frame()

    modeling = observed[
        [
            'trip_id', 'stop_id', 'route_short_name', 'delay_minutes', 'is_delayed',
            'observation_date_local', 'occupancy_label', 'current_status_name',
        ] + FEATURE_COLUMNS
    ].copy()

    modeling.to_csv(OUTPUTS_DIR / 'modeling_dataset.csv', index=False)

    summary = {
        'full_rows': int(len(full_df)),
        'observed_rows': int(len(observed)),
        'observed_delay_rate': float(observed['is_delayed'].mean()),
        'observed_routes': int(observed['route_short_name'].nunique()),
        'observed_stops': int(observed['stop_id'].nunique()),
        'observation_date_min': str(observed['observation_date_local'].dropna().min()),
        'observation_date_max': str(observed['observation_date_local'].dropna().max()),
    }

    with open(OUTPUTS_DIR / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved modeling_dataset.csv and dataset_summary.json')
    print(summary)


if __name__ == '__main__':
    main()
