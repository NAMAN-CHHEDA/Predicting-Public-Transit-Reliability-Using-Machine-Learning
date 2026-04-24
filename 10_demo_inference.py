from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "project_outputs" / "models" / "best_transit_delay_model.pkl"
DATA_PATH = ROOT / "data" / "cleaned" / "vta_features.csv"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(DATA_PATH)
    sample = df[feature_columns].head(5).copy()

    x_scaled = pd.DataFrame(scaler.transform(sample), columns=feature_columns, index=sample.index)
    preds = model.predict(x_scaled)
    probs = model.predict_proba(x_scaled)[:, 1]

    out = sample.copy()
    out["predicted_is_delayed"] = preds
    out["delay_probability"] = probs.round(4)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
