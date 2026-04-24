from __future__ import annotations

"""
Optional helper script.

Purpose:
Download NOAA Daily Summaries for the same date range as your realtime observations.
This lets you fix the weather/date mismatch in the original pipeline.

How to use:
1. Request a NOAA CDO token.
2. Export it in your shell:
   export NOAA_TOKEN=your_token_here
3. Run:
   python 6_fetch_noaa_weather.py

The script writes:
  outputs/noaa_observation_window_weather.csv
"""

import os
from datetime import datetime

import pandas as pd
import requests

from common import load_modeling_frame
from config import OUTPUTS_DIR

NOAA_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
DEFAULT_STATION = 'GHCND:USC00047821'
DEFAULT_DATASET = 'GHCND'
DEFAULT_DATATYPES = ['PRCP', 'TMAX', 'TMIN']


def main() -> None:
    token = os.getenv('NOAA_TOKEN')
    if not token:
        raise RuntimeError('NOAA_TOKEN environment variable is missing.')

    _, observed = load_modeling_frame()
    start_date = str(observed['observation_date_local'].dropna().min())
    end_date = str(observed['observation_date_local'].dropna().max())

    headers = {'token': token}
    rows = []

    for datatype in DEFAULT_DATATYPES:
        params = {
            'datasetid': DEFAULT_DATASET,
            'stationid': DEFAULT_STATION,
            'datatypeid': datatype,
            'startdate': start_date,
            'enddate': end_date,
            'limit': 1000,
            'units': 'standard',
        }
        response = requests.get(NOAA_URL, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get('results', []))

    weather = pd.DataFrame(rows)
    if weather.empty:
        raise RuntimeError('No weather rows returned from NOAA.')

    weather['date'] = pd.to_datetime(weather['date']).dt.date.astype(str)
    out = (
        weather.pivot_table(index='date', columns='datatype', values='value', aggfunc='first')
        .reset_index()
        .rename(columns={'date': 'observation_date_local'})
    )

    out.to_csv(OUTPUTS_DIR / 'noaa_observation_window_weather.csv', index=False)
    print(f'Saved weather rows for {start_date} through {end_date}')


if __name__ == '__main__':
    main()
