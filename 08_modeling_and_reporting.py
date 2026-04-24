import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned" / "vta_features.csv"
REPORT_DIR = ROOT / "project_outputs"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = REPORT_DIR / "models"
TABLE_DIR = REPORT_DIR / "tables"


def ensure_dirs() -> None:
    for folder in [REPORT_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    if "is_delayed" not in df.columns:
        raise ValueError("Expected target column 'is_delayed' in vta_features.csv")
    x = df.drop(columns=["is_delayed"]).copy()
    x = x.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    x = x.clip(lower=-1_000_000, upper=1_000_000)
    y = df["is_delayed"].astype(int)
    return x, y


def save_eda_visuals(df_full: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(x="is_delayed", data=df_full, hue="is_delayed", palette="Set2", legend=False)
    plt.title("Target Class Distribution")
    plt.xlabel("is_delayed")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png", dpi=180)
    plt.close()

    corr = df_full.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180, max_depth=8, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_models(
    models: dict[str, object], x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float], dict[str, list[float]]]:
    metrics_rows = []
    trained_models = {}
    probs_map = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
        }
        metrics_rows.append(row)
        trained_models[name] = model
        probs_map[name] = y_prob

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="f1", ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]
    return metrics_df, trained_models, {"best_model_name": best_model_name}, probs_map


def save_curve_plots(
    metrics_df: pd.DataFrame,
    probs_map: dict[str, list[float]],
    y_test: pd.Series,
    best_model_name: str,
    y_best_pred: pd.Series,
) -> None:
    plt.figure(figsize=(7, 5))
    for model_name in metrics_df["model"]:
        fpr, tpr, _ = roc_curve(y_test, probs_map[model_name])
        auc_val = metrics_df.loc[metrics_df["model"] == model_name, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for model_name in metrics_df["model"]:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs_map[model_name])
        pr_auc_val = metrics_df.loc[metrics_df["model"] == model_name, "pr_auc"].iloc[0]
        plt.plot(recall_vals, precision_vals, label=f"{model_name} (AP={pr_auc_val:.3f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curves.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_best_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_best_model.png", dpi=180)
    plt.close()


def save_feature_importance(best_model: object, feature_names: list[str]) -> None:
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        top10 = importances.head(10).reset_index()
        top10.columns = ["feature", "importance"]
        sns.barplot(data=top10, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance_top10.png", dpi=180)
        plt.close()

        importances.to_csv(TABLE_DIR / "feature_importance_full.csv", header=["importance"])


def main() -> None:
    ensure_dirs()
    x, y = load_data()
    df_full = x.copy()
    df_full["is_delayed"] = y
    save_eda_visuals(df_full)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x.columns, index=x_test.index)

    models = build_models()
    metrics_df, trained_models, details, probs_map = evaluate_models(
        models, x_train_scaled, x_test_scaled, y_train, y_test
    )
    best_model_name = details["best_model_name"]
    best_model = trained_models[best_model_name]

    y_best_pred = best_model.predict(x_test_scaled)
    y_best_prob = best_model.predict_proba(x_test_scaled)[:, 1]

    metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(classification_report(y_test, y_best_pred, output_dict=True)).T.to_csv(
        TABLE_DIR / "best_model_classification_report.csv"
    )

    save_curve_plots(metrics_df, probs_map, y_test, best_model_name, y_best_pred)
    save_feature_importance(best_model, x.columns.tolist())

    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_columns": x.columns.tolist(),
    }
    joblib.dump(bundle, MODEL_DIR / "best_transit_delay_model.pkl", compress=3)

    run_summary = {
        "dataset_rows": int(df_full.shape[0]),
        "dataset_columns": int(df_full.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "best_model_name": best_model_name,
        "best_model_f1": float(f1_score(y_test, y_best_pred)),
        "best_model_roc_auc": float(roc_auc_score(y_test, y_best_prob)),
        "best_model_pr_auc": float(average_precision_score(y_test, y_best_prob)),
    }
    with open(TABLE_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("Saved outputs in:")
    print(f"- {REPORT_DIR}")
    print(f"- {FIG_DIR}")
    print(f"- {TABLE_DIR}")
    print(f"- {MODEL_DIR / 'best_transit_delay_model.pkl'}")


if __name__ == "__main__":
    main()
