#!/usr/bin/env python3
"""
Predict CSV samples using saved artifacts (InfoGain-only + RF)

- Input: CSV berisi fitur numerik (boleh tanpa 'status').
- Output: CSV yang sama + kolom proba_phishing & prediction (0/1).
- Jika input mengandung 'status', script akan mencetak metrik evaluasi.
"""

import argparse, os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

ARTIFACTS = Path("artifacts")

def load_artifacts():
    model = joblib.load(ARTIFACTS / "model.pkl")
    selected = json.load(open(ARTIFACTS / "selected_features.json"))
    config = json.load(open(ARTIFACTS / "config.json"))
    kbins = None
    kbins_path = ARTIFACTS / "kbins.pkl"
    if kbins_path.exists():
        kbins = joblib.load(kbins_path)
    return model, selected, config, kbins

def preprocess_for_mode(df: pd.DataFrame, kbins, mode: str) -> pd.DataFrame:
    X = df.select_dtypes(include=[np.number]).copy()
    if mode == "discretized" and kbins is not None:
        cont_cols = [c for c in X.columns if not c.startswith("disc_") and X[c].nunique() > 10]
        if len(cont_cols) > 0:
            Xb = kbins.transform(X[cont_cols])
            Xb = pd.DataFrame(Xb, columns=[f"disc_{c}" for c in cont_cols], index=X.index)
            X = pd.concat([X.drop(columns=cont_cols), Xb], axis=1)
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV file with features")
    parser.add_argument("--output", type=str, required=True, help="CSV file to write predictions")
    args = parser.parse_args()

    model, selected, config, kbins = load_artifacts()
    df = pd.read_csv(args.input)

    has_label = "status" in df.columns
    if has_label:
        y_true = (df["status"] == "phishing").astype(int).values
    else:
        y_true = None

    X = preprocess_for_mode(df, kbins, config["mode"])

    # Pastikan semua fitur terpilih ada. Jika tidak ada di input, isi 0.0.
    for feat in selected:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[selected]

    proba = model.predict_proba(X)[:, 1]
    th = float(config.get("threshold", 0.5))
    pred = (proba >= th).astype(int)

    out = df.copy()
    out["proba_phishing"] = proba
    out["prediction"] = pred
    out.to_csv(args.output, index=False)
    print("Predictions saved to:", args.output)

    if has_label:
        acc = accuracy_score(y_true, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        print("\n=== EVAL on provided CSV ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1:        {f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_true, pred, digits=4))

if __name__ == "__main__":
    main()
