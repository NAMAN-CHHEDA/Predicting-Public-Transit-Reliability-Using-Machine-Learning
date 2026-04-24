from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common import FEATURE_COLUMNS, build_preprocessor, load_modeling_frame
from config import MODELS_DIR, OUTPUTS_DIR

SEED = 42


def score_values(pipeline: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps['model']
    if hasattr(model, 'predict_proba'):
        return pipeline.predict_proba(X_test)[:, 1]
    return pipeline.decision_function(X_test)


def evaluate_test_set(model_name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_score = score_values(pipeline, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        'model': model_name,
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'test_roc_auc': float(roc_auc_score(y_test, y_score)),
        'test_true_negatives': int(tn),
        'test_false_positives': int(fp),
        'test_false_negatives': int(fn),
        'test_true_positives': int(tp),
        'test_rows': int(len(y_test)),
        'predicted_positive_rows': int(y_pred.sum()),
    }


def main() -> None:
    _, observed = load_modeling_frame()
    X = observed[FEATURE_COLUMNS].copy()
    y = observed['is_delayed'].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }

    candidate_models = {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
        ),
        'svm_rbf': SVC(class_weight='balanced', kernel='rbf', random_state=SEED),
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
            'cv_accuracy_mean': float(cv_result['test_accuracy'].mean()),
            'cv_accuracy_std': float(cv_result['test_accuracy'].std()),
            'cv_precision_mean': float(cv_result['test_precision'].mean()),
            'cv_recall_mean': float(cv_result['test_recall'].mean()),
            'cv_f1_mean': float(cv_result['test_f1'].mean()),
            'cv_roc_auc_mean': float(cv_result['test_roc_auc'].mean()),
        }

        pipeline.fit(X_train, y_train)
        row.update(evaluate_test_set(model_name, pipeline, X_test, y_test))
        rows.append(row)
        fitted_pipelines[model_name] = pipeline

    metrics = pd.DataFrame(rows).sort_values(['cv_f1_mean', 'cv_recall_mean'], ascending=False)
    metrics.to_csv(OUTPUTS_DIR / 'classification_metrics.csv', index=False)

    best_model_name = metrics.iloc[0]['model']

    final_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', candidate_models[best_model_name]),
    ])
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODELS_DIR / 'delay_risk_classifier.joblib')

    preprocessor = final_pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()

    if best_model_name == 'logistic_regression':
        coef = final_pipeline.named_steps['model'].coef_[0]
        importance = pd.DataFrame({
            'feature': feature_names,
            'value': coef,
            'absolute_value': np.abs(coef),
        }).sort_values('absolute_value', ascending=False)
    elif best_model_name == 'random_forest':
        coef = final_pipeline.named_steps['model'].feature_importances_
        importance = pd.DataFrame({'feature': feature_names, 'value': coef}).sort_values('value', ascending=False)
    else:
        importance = pd.DataFrame({'feature': feature_names})

    importance.to_csv(OUTPUTS_DIR / 'classifier_feature_importance.csv', index=False)

    with open(OUTPUTS_DIR / 'classification_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'best_model': best_model_name,
                'full_observed_rows': int(len(observed)),
                'positive_rows': int(y.sum()),
                'negative_rows': int((1 - y).sum()),
            },
            f,
            indent=2,
        )

    print(metrics.to_string(index=False))
    print(f'Best classifier: {best_model_name}')


if __name__ == '__main__':
    main()
