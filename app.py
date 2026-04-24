from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from common import FEATURE_COLUMNS, load_modeling_frame
from config import MODELS_DIR, OUTPUTS_DIR

st.set_page_config(page_title='VTA Delay Risk Dashboard', layout='wide')

classifier = joblib.load(MODELS_DIR / 'delay_risk_classifier.joblib')
regressor = joblib.load(MODELS_DIR / 'delay_minutes_regressor.joblib')
_, observed = load_modeling_frame()
metrics_cls = pd.read_csv(OUTPUTS_DIR / 'classification_metrics.csv')
metrics_reg = pd.read_csv(OUTPUTS_DIR / 'regression_metrics.csv')
clusters = pd.read_csv(OUTPUTS_DIR / 'top_bottleneck_stops.csv')
feature_importance = pd.read_csv(OUTPUTS_DIR / 'classifier_feature_importance.csv')

route_defaults = (
    observed.groupby('route_short_name', observed=False)
    .agg(
        route_stop_count=('route_stop_count', 'median'),
        trip_length=('trip_length', 'median'),
        stop_sequence=('stop_sequence', 'median'),
        stop_position_pct=('stop_position_pct', 'median'),
        speed_mph=('speed_mph', 'median'),
        direction_id=('direction_id', 'median'),
    )
    .reset_index()
)
route_lookup = route_defaults.set_index('route_short_name')

st.title('VTA Delay Risk Dashboard')
st.caption('Traditional ML only: classification, regression, clustering, and explainability.')

left, right = st.columns([1.1, 1.1])

with left:
    st.subheader('Predict delay risk')
    route_name = st.selectbox('Route', sorted(observed['route_short_name'].dropna().unique().tolist()))
    default_row = route_lookup.loc[route_name]

    hour_of_day = st.slider('Scheduled hour of day', 0, 23, int(default_row.get('stop_sequence', 12) % 24))
    is_weekend = st.selectbox('Weekend?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_rush_hour = st.selectbox('Rush hour?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    direction_id = st.selectbox('Direction', [0, 1], index=int(default_row.get('direction_id', 0)))
    occupancy_label = st.selectbox('Occupancy', ['EMPTY', 'FEW_SEATS_AVAILABLE', 'FULL'])
    current_status_name = st.selectbox('Realtime status', ['INCOMING_AT', 'IN_TRANSIT_TO'])
    speed_mph = st.slider('Vehicle speed (mph)', 0.0, 60.0, float(default_row.get('speed_mph', 10.0)))
    stop_position_pct = st.slider('Stop position in trip', 0.0, 1.0, float(default_row.get('stop_position_pct', 0.5)))

    sample = pd.DataFrame([
        {
            'hour_of_day': hour_of_day,
            'is_rush_hour': is_rush_hour,
            'is_weekend': is_weekend,
            'route_stop_count': float(default_row['route_stop_count']),
            'trip_length': float(default_row['trip_length']),
            'stop_position_pct': stop_position_pct,
            'stop_sequence': max(1.0, round(float(default_row['trip_length']) * stop_position_pct)),
            'direction_id': direction_id,
            'speed_mph': speed_mph,
            'route_short_name': route_name,
            'occupancy_label': occupancy_label,
            'current_status_name': current_status_name,
        }
    ])[FEATURE_COLUMNS]

    risk = float(classifier.predict_proba(sample)[0, 1])
    delay_pred = float(regressor.predict(sample)[0])

    st.metric('Predicted delay risk', f'{risk:.1%}')
    st.metric('Predicted delay minutes', f'{delay_pred:.2f}')

with right:
    st.subheader('Current project results')
    st.write(metrics_cls[['model', 'cv_f1_mean', 'cv_recall_mean', 'cv_roc_auc_mean']])
    st.write(metrics_reg[['model', 'cv_mae_mean', 'cv_rmse_mean', 'cv_r2_mean']])

st.subheader('Top bottleneck stops')
st.dataframe(clusters[['stop_name', 'mean_late_minutes', 'late_rate', 'observations']].head(10))

st.subheader('Key drivers')
st.dataframe(feature_importance.head(15))
