from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR


def main() -> None:
    cls = pd.read_csv(OUTPUTS_DIR / 'classification_metrics.csv')
    reg = pd.read_csv(OUTPUTS_DIR / 'regression_metrics.csv')
    bottlenecks = pd.read_csv(OUTPUTS_DIR / 'top_bottleneck_stops.csv')

    fig = plt.figure(figsize=(8, 5))
    plt.bar(cls['model'], cls['cv_f1_mean'])
    plt.ylabel('Cross-validated F1')
    plt.title('Classifier Comparison')
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / 'classifier_comparison.png', dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.bar(reg['model'], reg['cv_mae_mean'])
    plt.ylabel('Cross-validated MAE (minutes)')
    plt.title('Regression Comparison')
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / 'regression_comparison.png', dpi=200)
    plt.close(fig)

    top10 = bottlenecks.head(10).copy()
    labels = top10['stop_name'].fillna(top10['stop_id']).astype(str)
    fig = plt.figure(figsize=(9, 6))
    plt.barh(labels, top10['mean_late_minutes'])
    plt.xlabel('Average late minutes')
    plt.title('Top Bottleneck Stops')
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / 'top_bottleneck_stops.png', dpi=200)
    plt.close(fig)

    print('Saved PNG figures into outputs/.')


if __name__ == '__main__':
    main()
