#!/usr/bin/env python3
"""
PREDICT PHISHING - RAW FEATURES ONLY
Compatible with Streamlit
"""

import argparse, json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

ARTIFACTS = Path("artifacts")

def load_artifacts():
    model = joblib.load(ARTIFACTS / "model.pkl")
    selected = json.load(open(ARTIFACTS / "selected_features.json"))
    return model, selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model, selected = load_artifacts()

    df = pd.read_csv(args.input)
    has_label = "status" in df.columns

    if has_label:
        y_true = (df["status"] == "phishing").astype(int).values

    X = df.select_dtypes(include=[np.number]).copy()

    # ALIGN FEATURES
    for f in selected:
        if f not in X.columns:
            X[f] = 0.0

    X = X[selected]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["proba_phishing"] = proba
    out["prediction"] = pred
    out.to_csv(args.output, index=False)

    print("Prediction saved to:", args.output)

    if has_label:
        print("\n=== EVALUATION ===")
        print(classification_report(y_true, pred))

if __name__ == "__main__":
    main()
