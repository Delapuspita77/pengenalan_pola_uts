#!/usr/bin/env python3
"""
TRAIN MODEL PHISHING - RAW FEATURES ONLY
InfoGain (Top-K) + RandomForest
"""

import argparse, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    if "label" in df.columns and "status" not in df.columns:
        df.rename(columns={"label": "status"}, inplace=True)
    df = df[df["status"].isin(["legitimate", "phishing"])].copy()
    if "url" in df.columns:
        df.drop(columns=["url"], inplace=True)
    return df.reset_index(drop=True)

def split_xy(df):
    y = (df["status"] == "phishing").astype(int).values
    X = df.drop(columns=["status"]).select_dtypes(include=[np.number]).copy()
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset phising raw.csv")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # LOAD DATA
    df = load_and_clean(args.data)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # INFO GAIN (RAW)
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    mi = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    selected = mi.head(args.top_k).index.tolist()

    print("\nTOP FEATURES:")
    for f in selected:
        print("-", f)

    # MODEL
    base = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train[selected], y_train)

    # EVALUATION
    proba = model.predict_proba(X_test[selected])[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("\n=== EVALUATION ===")
    print(classification_report(y_test, pred))

    # SAVE ARTIFACTS
    joblib.dump(model, ARTIFACTS / "model.pkl")
    json.dump(selected, open(ARTIFACTS / "selected_features.json", "w"), indent=2)

    print("\nTRAIN SELESAI (RAW MODE)")
    print("Artifacts saved in ./artifacts")

if __name__ == "__main__":
    main()
