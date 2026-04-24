# Predicting Public Transit Reliability Using Traditional Machine Learning

This package turns your current DATA 245 transit project into a full end-to-end beginner-friendly workflow:

1. build the modeling dataset from your current `vta_final.csv`
2. train a delay-risk classifier
3. train a delay-minutes regressor
4. cluster stops to identify bottlenecks
5. generate presentation-ready figures
6. run a simple Streamlit frontend

## Important project reality

Your current merged dataset is large, but only a small part of it has observed realtime delay labels.
That means the project is valid, but the present results should be described as a **prototype / proof-of-concept** unless you collect more GTFS-RT snapshots.

A second issue in the original code is that weather was merged using the GTFS `start_date` service window instead of the actual realtime observation date. That is why the scripts here focus on a solid ML pipeline first, while recommending a weather-alignment fix before your final submission.

## Folder assumption

These scripts assume your extracted class project lives at:

`/mnt/data/data245_unzip/Data 245 Project/Predicting-Public-Transit-Reliability-Using-Machine-Learning`

If your local folder is different, edit `config.py`.

## Step-by-step run order

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare modeling dataset

```bash
python 1_prepare_modeling_data.py
```

### 3) Train classifier

```bash
python 2_train_classifier.py
```

### 4) Train regressor

```bash
python 3_train_regressor.py
```

### 5) Cluster bottleneck stops

```bash
python 4_cluster_stops.py
```

### 6) Make figures

```bash
python 5_make_figures.py
```

### 7) Run frontend

```bash
streamlit run app.py
```

## What to say in your final report

### Problem statement
Predict whether a VTA bus stop event will be delayed by more than 5 minutes, estimate how many minutes late it may be, and identify the operational factors and locations associated with unreliability.

### Models implemented
- Logistic Regression
- Random Forest Classifier
- SVM Classifier
- Linear Regression
- Random Forest Regressor
- K-Means clustering for stop bottlenecks

### Why this is course-aligned
- traditional ML only
- interpretable pipeline
- feature engineering from tabular data
- model comparison
- classification + regression + clustering
- end-user dashboard

## Suggested final presentation structure

1. Motivation and why transit reliability matters
2. Data sources: GTFS static, GTFS-realtime, NOAA weather
3. Data engineering pipeline
4. Class imbalance challenge
5. Classification results
6. Regression results
7. Bottleneck stop clustering
8. Key drivers of unreliability
9. Dashboard demo
10. Limitations and future work

## Strong limitations to mention honestly

- only a small number of stop events have observed delay labels
- current weather alignment in the original pipeline needs correction
- vehicle-position timestamps approximate delay and are not the same as exact stop arrival confirmations
- more realtime snapshots will improve model stability

## Best next upgrade

Collect more `vehicle_positions_*.csv` and `realtime_snapshot_*.csv` files over at least 2 to 4 weeks, then rerun the training scripts.
