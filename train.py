#!/usr/bin/env python3
"""
Train model: InfoGain-only (Top-K) + RandomForest (classifier only)

- Input dataset: CSV dengan kolom 'status' (legitimate/phishing) + fitur numerik.
- Seleksi fitur: mutual_info_classif (Top-K).
- Mode:
  - raw          -> InfoGain langsung di fitur numerik
  - discretized  -> fitur kontinu dibinning dulu (KBins) lalu InfoGain dengan discrete_features=True
- Output artefak (disimpan ke ./artifacts/):
  - model.pkl                 (RandomForest + probabilitas terkalibrasi)
  - kbins.pkl                 (jika mode=discretized)
  - selected_features.json    (daftar Top-K)
  - feature_schema.json
  - config.json               (mode, top_k, threshold, seed)
"""

import argparse, os, json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore", category=UserWarning)

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def load_and_clean(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "status" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "status"})
    if "status" not in df.columns:
        raise KeyError("Expected a 'status' (or 'label') column.")
    df = df[df["status"].isin(["legitimate", "phishing"])].copy()
    if "url" in df.columns:
        df.drop(columns=["url"], inplace=True)
    return df.reset_index(drop=True)

def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    y = (df["status"] == "phishing").astype(int).values
    X = df.drop(columns=["status"]).select_dtypes(include=[np.number]).copy()
    return X, y

def discretize_if_needed(X_train: pd.DataFrame, X_val: pd.DataFrame, mode: str):
    kbins = None
    X_train_p = X_train.copy()
    X_val_p = X_val.copy()
    if mode == "discretized":
        cont_cols = [c for c in X_train.columns if X_train[c].nunique() > 10]
        if len(cont_cols) > 0:
            kbins = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
            Xb_tr = kbins.fit_transform(X_train_p[cont_cols])
            Xb_va = kbins.transform(X_val_p[cont_cols])
            Xb_tr = pd.DataFrame(Xb_tr, columns=[f"disc_{c}" for c in cont_cols], index=X_train_p.index)
            Xb_va = pd.DataFrame(Xb_va, columns=[f"disc_{c}" for c in cont_cols], index=X_val_p.index)
            X_train_p = pd.concat([X_train_p.drop(columns=cont_cols), Xb_tr], axis=1)
            X_val_p   = pd.concat([X_val_p.drop(columns=cont_cols),   Xb_va], axis=1)
    return X_train_p, X_val_p, kbins

def select_topk_infogain(X: pd.DataFrame, y: np.ndarray, top_k: int, mode: str):
    if mode == "discretized":
        mi = mutual_info_classif(X, y, discrete_features=True, random_state=42)
    else:
        mi = mutual_info_classif(X, y, random_state=42)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi_s.head(top_k).index.tolist()

def pick_threshold_by_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 19):
        pred = (proba >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/dataset phising raw.csv",
                        help="Path CSV dataset (default: data/dataset phising raw.csv)")
    parser.add_argument("--mode", type=str, choices=["raw","discretized"], default="discretized")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1) Load & split
    df = load_and_clean(args.data)
    X, y = split_xy(df)

    # 2) Split train/holdout -> train/val
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=args.seed, stratify=y_train)

    # 3) Optional discretization
    X_tr_p, X_val_p, kbins = discretize_if_needed(X_tr, X_val, args.mode)

    # 4) Feature selection (Top-K)
    selected = select_topk_infogain(X_tr_p, y_tr, args.top_k, args.mode)

    # 5) Model: RF + calibration for probabilities
    base_rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=args.seed)
    clf = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)
    clf.fit(X_tr_p[selected], y_tr)

    # 6) Threshold by F1 (on validation)
    proba_val = clf.predict_proba(X_val_p[selected])[:, 1]
    th = pick_threshold_by_f1(y_val, proba_val)

    # 7) Evaluate on holdout (apply same preprocessing)
    X_tr2, X_hold2 = X_train.copy(), X_hold.copy()
    if args.mode == "discretized" and kbins is not None:
        cont_cols = [c for c in X_train.columns if X_train[c].nunique() > 10]
        Xb_tr = kbins.transform(X_tr2[cont_cols])
        Xb_ho = kbins.transform(X_hold2[cont_cols])
        Xb_tr = pd.DataFrame(Xb_tr, columns=[f"disc_{c}" for c in cont_cols], index=X_tr2.index)
        Xb_ho = pd.DataFrame(Xb_ho, columns=[f"disc_{c}" for c in cont_cols], index=X_hold2.index)
        X_tr2 = pd.concat([X_tr2.drop(columns=cont_cols), Xb_tr], axis=1)
        X_hold2 = pd.concat([X_hold2.drop(columns=cont_cols), Xb_ho], axis=1)

    proba_hold = clf.predict_proba(X_hold2[selected])[:, 1]
    pred_hold = (proba_hold >= th).astype(int)

    acc = accuracy_score(y_hold, pred_hold)
    p, r, f1, _ = precision_recall_fscore_support(y_hold, pred_hold, average="binary", zero_division=0)

    print("\n=== HOLDOUT METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_hold, pred_hold, digits=4))

    # 8) Save artifacts
    joblib.dump(clf, ARTIFACTS / "model.pkl")
    if kbins is not None:
        joblib.dump(kbins, ARTIFACTS / "kbins.pkl")
    with open(ARTIFACTS / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(ARTIFACTS / "feature_schema.json", "w") as f:
        json.dump({"numeric_features": list(X.columns), "label": "status"}, f, indent=2)
    with open(ARTIFACTS / "config.json", "w") as f:
        json.dump({"mode": args.mode, "top_k": args.top_k, "threshold": th, "seed": args.seed}, f, indent=2)

    print("\nArtifacts saved in:", ARTIFACTS.resolve())
    print("Selected features (Top-{}): {}".format(args.top_k, selected))
    print("Chosen threshold (max F1 on val): {:.3f}".format(th))

if __name__ == "__main__":
    main()
