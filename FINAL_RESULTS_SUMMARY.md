# Current Results Summary

## Data coverage
- Total rows in merged dataset: 361,885
- Rows with observed realtime delay labels: 591
- Delayed rows (> 5 minutes late): 48
- Observed route count: 33
- Observed stop count: 402
- Realtime observation window: 2026-04-05 to 2026-04-23

## Best classification model
**Logistic Regression** was selected as the best practical classifier on the current data because it had the strongest balance of recall and F1 while staying interpretable.

### Cross-validated classification performance
- Accuracy: 0.699
- Precision: 0.155
- Recall: 0.604
- F1: 0.245
- ROC-AUC: 0.692

### Held-out test performance
- Accuracy: 0.709
- Precision: 0.170
- Recall: 0.667
- F1: 0.271
- ROC-AUC: 0.725

## Best regression model
**Random Forest Regressor** was selected as the best delay-minutes regressor on the current data.

### Cross-validated regression performance
- MAE: 4.276 minutes
- RMSE: 9.518 minutes
- R^2: -0.129

### Held-out test performance
- MAE: 4.937 minutes
- RMSE: 17.612 minutes
- R^2: 0.080

## Key drivers of unreliability
From the trained models, the strongest signal variables were:
- route identity
- whether the trip was on a weekend
- hour of day
- rush-hour flag
- stop position within the trip
- stop sequence
- vehicle speed
- occupancy level

## Highest-risk bottleneck stops in the current sample
1. El Camino & Castro
2. Showers & California
3. El Camino & San Antonio
4. The Alameda & Portola
5. Bascom & Fruitdale
6. El Camino & Showers
7. Santa Clara & 24th
8. Main & Great Mall Pkwy
9. El Camino & Jordan
10. Stelling & Stevens Creek

## Honest limitation to say in the report
The current project is a strong prototype, but not yet a high-confidence production predictor, because only 591 stop events currently have observed delay labels and only 48 of those are delayed. Additional GTFS-realtime snapshots collected over more days will make the model more stable and improve the credibility of the final analysis.

## Important pipeline correction
The original project code merged weather by GTFS `start_date`, which does not match the realtime observation date. Use `6_fetch_noaa_weather.py` and then merge weather by `observation_date_local` before final submission if you want weather-based conclusions to be fully defensible.
