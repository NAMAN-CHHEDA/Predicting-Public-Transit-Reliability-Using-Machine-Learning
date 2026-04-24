from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from common import FEATURE_COLUMNS, build_preprocessor, load_modeling_frame
from config import MODELS_DIR, OUTPUTS_DIR


SEED = 42


def main() -> None:
    _, observed = load_modeling_frame()
    X = observed[FEATURE_COLUMNS].copy()
    y = observed['delay_minutes'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
    }

    candidate_models = {
        'linear_regression': LinearRegression(),
        'random_forest_regressor': RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    rows = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor()),
            ('model', estimator),
        ])

        cv_result = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            'model': model_name,
            'cv_mae_mean': float(-cv_result['test_mae'].mean()),
            'cv_rmse_mean': float(-cv_result['test_rmse'].mean()),
            'cv_r2_mean': float(cv_result['test_r2'].mean()),
        }

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_mae = float((y_test - y_pred).abs().mean())
        test_rmse = float((((y_test - y_pred) ** 2).mean()) ** 0.5)
        sst = float(((y_test - y_test.mean()) ** 2).sum())
        sse = float(((y_test - y_pred) ** 2).sum())
        test_r2 = float(1 - sse / sst) if sst != 0 else 0.0

        row.update(
            {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_rows': int(len(y_test)),
            }
        )
        rows.append(row)
        fitted_pipelines[model_name] = pipeline

    metrics = pd.DataFrame(rows).sort_values(['cv_mae_mean', 'cv_rmse_mean'], ascending=True)
    metrics.to_csv(OUTPUTS_DIR / 'regression_metrics.csv', index=False)

    best_model_name = metrics.iloc[0]['model']
    final_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', candidate_models[best_model_name]),
    ])
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODELS_DIR / 'delay_minutes_regressor.joblib')

    with open(OUTPUTS_DIR / 'regression_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'best_model': best_model_name,
                'full_observed_rows': int(len(observed)),
                'delay_min': float(observed['delay_minutes'].min()),
                'delay_max': float(observed['delay_minutes'].max()),
            },
            f,
            indent=2,
        )

    print(metrics.to_string(index=False))
    print(f'Best regressor: {best_model_name}')


if __name__ == '__main__':
    main()
