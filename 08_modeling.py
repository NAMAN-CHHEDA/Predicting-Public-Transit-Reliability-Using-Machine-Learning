
"""
08_modeling_naeem_final.py

Final rewritten modeling script for Naeem's modeling task.

Purpose:
Train traditional machine learning models to predict whether a VTA transit stop
event is delayed using engineered features, while avoiding direct and proxy leakage.

Important improvements over earlier versions:
1. Removes obvious leakage columns:
   - delay_minutes
   - actual_delay_seconds
   - delay_vehicle_id
   - any column containing "delay" except target is_delayed
2. Removes proxy leakage columns:
   - precise arrival/departure timedelta features
   - precise schedule seconds columns
   - start/end/wx date derived features if they behave as proxies
3. Adds a DummyClassifier baseline.
4. Uses Stratified K-Fold cross-validation.
5. Uses SMOTE inside each training fold if imbalanced-learn is installed.
6. Uses imbalanced-classification metrics:
   - F1
   - precision
   - recall
   - ROC-AUC
   - average precision
7. Saves professor-ready outputs:
   - model_results_cv_summary_final.csv
   - model_results_cv_by_fold_final.csv
   - model_comparison_f1_final.png
   - confusion_matrix_<model>_final.png
   - roc_curve_<model>_final.png
   - precision_recall_curve_<model>_final.png
   - best_model_final.joblib
   - leakage_correlation_report_final.csv

Run:
    python 08_modeling_naeem_final.py

Recommended install:
    python -m pip install pandas numpy scikit-learn matplotlib joblib pyarrow imbalanced-learn

Optional:
    python -m pip install xgboost shap
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    PrecisionRecallDisplay,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
NEGATIVE_TO_POSITIVE_RATIO = 20
SMOTE_K_NEIGHBORS = 3


# ---------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------
def load_final_dataset() -> pd.DataFrame:
    """Load final dataset from parquet or CSV."""
    parquet_path = DATA_DIR / "vta_final.parquet"
    csv_path = DATA_DIR / "vta_final.csv"

    if parquet_path.exists():
        print(f"Loading parquet: {parquet_path}")
        try:
            return pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"Could not read parquet ({exc}). Trying CSV fallback...")

    if csv_path.exists():
        print(f"Loading CSV: {csv_path}")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "Could not find data/cleaned/vta_final.parquet or data/cleaned/vta_final.csv"
    )


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean target column."""
    if "is_delayed" not in df.columns:
        raise ValueError("Missing target column: is_delayed")

    df = df.copy()
    df = df.dropna(subset=["is_delayed"])

    if df["is_delayed"].dtype == "object":
        mapping = {
            "true": 1,
            "yes": 1,
            "delayed": 1,
            "1": 1,
            "false": 0,
            "no": 0,
            "not delayed": 0,
            "0": 0,
        }
        mapped = df["is_delayed"].astype(str).str.strip().str.lower().map(mapping)
        df["is_delayed"] = mapped.fillna(df["is_delayed"])

    df["is_delayed"] = df["is_delayed"].astype(int)
    return df


def reduce_majority_for_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps all delayed records and samples non-delayed records for runtime.

    This is not the final class-balancing method. It only makes training manageable.
    SMOTE, if installed, is applied later inside each cross-validation training fold.
    """
    counts = df["is_delayed"].value_counts()

    print("\nOriginal target counts:")
    print(counts)

    pos_df = df[df["is_delayed"] == 1]
    neg_df = df[df["is_delayed"] == 0]

    if len(pos_df) == 0:
        raise ValueError("No delayed rows found.")

    n_neg_keep = min(len(neg_df), max(500, len(pos_df) * NEGATIVE_TO_POSITIVE_RATIO))
    neg_sample = neg_df.sample(n=n_neg_keep, random_state=RANDOM_STATE)

    sampled = pd.concat([pos_df, neg_sample], axis=0)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nRuntime-sampled target counts:")
    print(sampled["is_delayed"].value_counts())

    return sampled


# ---------------------------------------------------------------------
# Feature engineering / leakage control
# ---------------------------------------------------------------------
def engineer_safe_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely converts broad date fields into coarse time features.

    NOTE:
    We intentionally avoid precise arrival/departure seconds and timedelta values
    because they can behave like hidden delay indicators.
    """
    df = df.copy()

    # Convert object date-like columns only if they are broad date fields.
    broad_date_cols = [
        c
        for c in df.columns
        if c.lower() in {"start_date", "end_date", "wx_date", "service_date", "date"}
    ]

    for col in broad_date_cols:
        if df[col].dtype == "object":
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().mean() > 0.50:
                df[col] = converted

    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    for col in datetime_cols:
        # Keep only coarse features. No exact timestamps.
        print(f"Converting broad datetime column: {col}")
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_month"] = df[col].dt.month
        df = df.drop(columns=[col])

    # Do NOT convert timedelta to seconds for model features.
    # Timedelta values like arrival_td or departure_td may be proxy leakage.
    timedelta_cols = df.select_dtypes(include=["timedelta64[ns]"]).columns.tolist()
    if timedelta_cols:
        print("Dropping timedelta columns as proxy leakage:", timedelta_cols)
        df = df.drop(columns=timedelta_cols)

    return df


def get_strict_drop_columns(df: pd.DataFrame) -> list[str]:
    """
    Build strict drop list for direct leakage, proxy leakage, and identifiers.
    """
    # Drop any delay-related feature except target.
    delay_like_cols = [
        c for c in df.columns
        if "delay" in c.lower() and c != "is_delayed"
    ]

    # Drop precise timing columns or schedule columns that can proxy delay.
    time_proxy_keywords = [
        "arrival",
        "departure",
        "timestamp",
        "timepoint",
        "_td",
        "td_",
        "_sec",
        "seconds",
    ]

    time_proxy_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in time_proxy_keywords)
    ]

    # IDs and high-cardinality identifiers.
    id_cols = [
        c for c in df.columns
        if c.lower() in {
            "trip_id",
            "vehicle_id",
            "shape_id",
            "block_id",
            "delay_vehicle_id",
            "delay_snapshot_ts",
        }
    ]

    explicit_cols = [
        "is_delayed",
        "delay_minutes",
        "actual_delay_seconds",
        "delay_vehicle_id",
        "stop_sequence",
        "timepoint",
        "trip_id",
        "vehicle_id",
        "shape_id",
        "block_id",
        "arrival_td",
        "departure_td",
        "arrival_td_seconds",
        "departure_td_seconds",
        "arrival_sec",
        "departure_sec",
    ]

    return sorted(set(delay_like_cols + time_proxy_cols + id_cols + explicit_cols))


def save_leakage_report(df: pd.DataFrame, filename: str = "leakage_correlation_report_final.csv") -> pd.DataFrame:
    """Save numeric correlation report with target."""
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()

    if "is_delayed" not in numeric_df.columns:
        return pd.DataFrame()

    rows = []
    for col in numeric_df.columns:
        if col == "is_delayed":
            continue
        if numeric_df[col].nunique(dropna=True) <= 1:
            continue
        try:
            corr = numeric_df[col].corr(numeric_df["is_delayed"])
            rows.append((col, corr, abs(corr)))
        except Exception:
            pass

    report = pd.DataFrame(rows, columns=["feature", "corr_with_target", "abs_corr"])
    report = report.sort_values("abs_corr", ascending=False)
    report.to_csv(OUTPUT_DIR / filename, index=False)

    print("\nTop remaining high-signal numeric columns after final leakage removal:")
    print(report.head(15))

    return report


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y with strict leakage removal."""
    df = engineer_safe_time_features(df)

    drop_cols = get_strict_drop_columns(df)

    print("\nDropping leakage/proxy/ID columns:")
    print([c for c in drop_cols if c in df.columns])

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["is_delayed"]

    # Drop remaining datetime/timedelta columns defensively.
    bad_cols = [
        c
        for c in X.columns
        if str(X[c].dtype).startswith("datetime")
        or str(X[c].dtype).startswith("timedelta")
    ]

    if bad_cols:
        print("Dropping remaining unsupported datetime/timedelta columns:", bad_cols)
        X = X.drop(columns=bad_cols)

    # Keep only sklearn-friendly dtypes.
    X = X.select_dtypes(include=["number", "bool", "object", "category"])

    # Convert object features to strings for OneHotEncoder.
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype(str)

    # Drop all-null columns.
    all_null_cols = X.columns[X.isna().all()].tolist()
    if all_null_cols:
        print("Dropping all-null columns:", all_null_cols)
        X = X.drop(columns=all_null_cols)

    print("\nFeature dtype counts:")
    print(X.dtypes.value_counts())
    print("Feature matrix shape:", X.shape)

    tmp = X.copy()
    tmp["is_delayed"] = y.values
    save_leakage_report(tmp)

    return X, y


# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical variables."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def get_models() -> dict:
    """Models for comparison."""
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def make_model_pipeline(preprocessor: ColumnTransformer, model, model_name: str):
    """
    Create model pipeline.

    SMOTE is applied only inside training folds and not to the baseline.
    """
    if IMBLEARN_AVAILABLE and model_name != "Dummy Baseline":
        return ImbPipeline(
            [
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
                ("model", model),
            ]
        )

    return SkPipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(name: str, pipeline, X_test, y_test) -> dict:
    """Evaluate predictions for one fold."""
    y_pred = pipeline.predict(X_test)

    roc_auc = np.nan
    avg_precision = np.nan

    model = pipeline.named_steps["model"]

    if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
        try:
            y_score = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            avg_precision = average_precision_score(y_test, y_score)
        except Exception:
            pass

    return {
        "model": name,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run Stratified K-Fold CV."""
    if IMBLEARN_AVAILABLE:
        print("\nSMOTE enabled inside each training fold for non-baseline models.\n")
    else:
        print("\nWARNING: imbalanced-learn is not installed. Running without SMOTE.")
        print("Install with: python -m pip install imbalanced-learn\n")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for model_name, model in get_models().items():
        print(f"\nCross-validating {model_name}...")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            preprocessor = make_preprocessor(X_train)
            pipeline = make_model_pipeline(preprocessor, model, model_name)

            try:
                pipeline.fit(X_train, y_train)
                metrics = evaluate_predictions(model_name, pipeline, X_test, y_test)
                metrics["fold"] = fold
                rows.append(metrics)

                print(
                    f"  Fold {fold}: "
                    f"F1={metrics['f1']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"Precision={metrics['precision']:.3f}"
                )
            except Exception as exc:
                print(f"  Fold {fold} failed: {exc}")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUTPUT_DIR / "model_results_cv_by_fold_final.csv", index=False)

    summary = (
        fold_df.groupby("model")
        .agg(
            {
                "f1": ["mean", "std"],
                "roc_auc": ["mean", "std"],
                "average_precision": ["mean", "std"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
            }
        )
    )

    summary.columns = ["_".join(c).strip() for c in summary.columns]
    summary = summary.reset_index().sort_values("f1_mean", ascending=False)
    summary.to_csv(OUTPUT_DIR / "model_results_cv_summary_final.csv", index=False)

    print("\nCross-validation summary:")
    print(summary)

    return summary


# ---------------------------------------------------------------------
# Output plots / final model
# ---------------------------------------------------------------------
def make_model_comparison_plot(summary: pd.DataFrame) -> None:
    """Save model comparison bar chart."""
    plt.figure(figsize=(9, 5))
    plt.bar(summary["model"], summary["f1_mean"])
    plt.ylabel("Mean F1 Score")
    plt.title("Cross-Validated Model Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_f1_final.png", dpi=200)
    plt.close()


def train_final_best_model(X: pd.DataFrame, y: pd.Series, best_model_name: str):
    """Train final model and save professor-ready plots."""
    models = get_models()
    model = models[best_model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(X_train)
    pipeline = make_model_pipeline(preprocessor, model, best_model_name)

    print(f"\nTraining final best model: {best_model_name}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nFinal holdout classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nSanity check: positive predictions in holdout set:")
    print(f"{int((y_pred == 1).sum())} positive predictions out of {len(y_pred)} rows")

    safe_name = best_model_name.lower().replace(" ", "_")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Delayed", "Delayed"],
    )
    disp.ax_.set_title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{safe_name}_final.png", dpi=200)
    plt.close()

    if hasattr(pipeline.named_steps["model"], "predict_proba") and best_model_name != "Dummy Baseline":
        y_score = pipeline.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(y_test, y_score)
        plt.title(f"ROC Curve - {best_model_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"roc_curve_{safe_name}_final.png", dpi=200)
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_score)
        plt.title(f"Precision-Recall Curve - {best_model_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"precision_recall_curve_{safe_name}_final.png", dpi=200)
        plt.close()

    joblib.dump(pipeline, OUTPUT_DIR / "best_model_final.joblib")
    print("Saved final model: outputs/best_model_final.joblib")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    df = load_final_dataset()
    df = clean_target(df)

    print("\nDataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["is_delayed"].value_counts(normalize=True))

    df_model = reduce_majority_for_runtime(df)
    X, y = build_feature_matrix(df_model)

    summary = cross_validate_models(X, y)
    make_model_comparison_plot(summary)

    if summary.empty:
        raise RuntimeError("No model results were produced.")

    # Do not select dummy baseline as best model, even if results tie.
    non_baseline = summary[summary["model"] != "Dummy Baseline"].copy()

    if non_baseline.empty:
        best_model_name = summary.iloc[0]["model"]
    else:
        best_model_name = non_baseline.iloc[0]["model"]

    train_final_best_model(X, y, best_model_name)

    print("\nDone.")
    print("Best non-baseline model:", best_model_name)
    print("Outputs saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
