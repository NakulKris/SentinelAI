"""
08_retrain_augmented.py — SentinelAI Retrain on Augmented Dataset
==================================================================
Trains a new TF-IDF + Logistic Regression model on the augmented v2 dataset
and compares metrics against the original v1 model.

Run from project root:
    python code/08_retrain_augmented.py
"""

import os, sys, json, time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")

MASTER_V2     = os.path.join(PROCESSED_DIR, "SentinelAI_Master_Dataset_v2.csv")
PREP_V1       = os.path.join(PROCESSED_DIR, "SentinelAI_Preprocessed_Dataset.csv")

MODEL_V2      = os.path.join(MODELS_DIR, "logistic_regression_v2.joblib")
VEC_V2        = os.path.join(MODELS_DIR, "tfidf_vectorizer_v2.joblib")
EVAL_V2       = os.path.join(MODELS_DIR, "evaluation_results_v2.json")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))

RANDOM_STATE = 42
TEST_FRAC    = 0.20


def main():
    t_total = time.perf_counter()

    # ── Load & preprocess ─────────────────────────────────────────────────
    print("\n[Retrain] Loading augmented dataset …")
    df = pd.read_csv(MASTER_V2, usecols=["text","label"], low_memory=False)
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    print(f"  Total rows : {len(df):,}")
    print(f"  Malicious  : {(df['label']==1).sum():,}")
    print(f"  Benign     : {(df['label']==0).sum():,}")

    print("\n[Retrain] Preprocessing text …")
    from preprocess import preprocess_text
    t0 = time.perf_counter()
    df["processed_text"] = df["text"].apply(preprocess_text)
    print(f"  Preprocessing done in {time.perf_counter()-t0:.1f}s")

    # Drop empty processed texts
    before = len(df)
    df = df[df["processed_text"].str.strip().str.len() > 0].copy()
    print(f"  Dropped {before - len(df)} empty rows after preprocessing")
    print(f"  ML-ready rows: {len(df):,}")

    # ── Leakage-free train/test split (group by processed_text) ──────────
    print("\n[Retrain] Creating leakage-free train/test split …")
    unique_texts = df["processed_text"].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_texts)
    n_test  = int(len(unique_texts) * TEST_FRAC)
    test_texts  = set(unique_texts[:n_test])
    train_texts = set(unique_texts[n_test:])

    df_train = df[df["processed_text"].isin(train_texts)].copy()
    df_test  = df[df["processed_text"].isin(test_texts)].copy()

    overlap = set(df_train["processed_text"]) & set(df_test["processed_text"])
    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}  |  Leakage overlap: {len(overlap)}")

    # ── TF-IDF ────────────────────────────────────────────────────────────
    print("\n[Retrain] Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        max_features  = 50000,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 2,
        max_df        = 0.95,
        strip_accents = "unicode",
        analyzer      = "word",
        token_pattern = r"\b[a-zA-Z]\w+\b",
    )
    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(df_train["processed_text"])
    X_test  = vectorizer.transform(df_test["processed_text"])
    print(f"  TF-IDF done in {time.perf_counter()-t0:.1f}s  |  Features: {X_train.shape[1]:,}")

    # ── Logistic Regression ───────────────────────────────────────────────
    print("\n[Retrain] Training Logistic Regression …")
    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="saga",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(X_train, df_train["label"])
    print(f"  Training done in {time.perf_counter()-t0:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[Retrain] Evaluating …")
    y_test  = df_test["label"]
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred),   6)
    prec = round(precision_score(y_test, y_pred),  6)
    rec  = round(recall_score(y_test, y_pred),     6)
    f1   = round(f1_score(y_test, y_pred),         6)
    auc  = round(roc_auc_score(y_test, y_proba),   6)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  V2 Model Results (Augmented Dataset)        │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  Accuracy  : {acc*100:.4f}%                       │")
    print(f"  │  Precision : {prec*100:.4f}%                       │")
    print(f"  │  Recall    : {rec*100:.4f}%                       │")
    print(f"  │  F1-Score  : {f1*100:.4f}%                       │")
    print(f"  │  ROC-AUC   : {auc*100:.4f}%                       │")
    print(f"  └──────────────────────────────────────────────┘")

    # Compare with v1
    v1_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"\n  ── Comparison vs V1 baseline ──")
        print(f"  Accuracy  : {v1['accuracy']*100:.4f}% → {acc*100:.4f}%  ({(acc-v1['accuracy'])*100:+.4f}%)")
        print(f"  Precision : {v1['precision']*100:.4f}% → {prec*100:.4f}%  ({(prec-v1['precision'])*100:+.4f}%)")
        print(f"  Recall    : {v1['recall']*100:.4f}% → {rec*100:.4f}%  ({(rec-v1['recall'])*100:+.4f}%)")
        print(f"  F1-Score  : {v1['f1_score']*100:.4f}% → {f1*100:.4f}%  ({(f1-v1['f1_score'])*100:+.4f}%)")
        print(f"  ROC-AUC   : {v1['roc_auc']*100:.4f}% → {auc*100:.4f}%  ({(auc-v1['roc_auc'])*100:+.4f}%)")

    # ── Save artifacts ────────────────────────────────────────────────────
    joblib.dump(model,      MODEL_V2)
    joblib.dump(vectorizer, VEC_V2)

    results = {
        "model": "Logistic Regression v2 (Augmented Dataset)",
        "vectorizer": "TF-IDF", "train_rows": len(df_train),
        "test_rows": len(df_test), "random_state": RANDOM_STATE,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "roc_auc": auc,
        "confusion_matrix": cm,
        "elapsed_seconds": round(time.perf_counter() - t_total, 1),
    }
    with open(EVAL_V2, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Retrain] Saved model → {MODEL_V2}")
    print(f"[Retrain] Saved vec   → {VEC_V2}")
    print(f"[Retrain] Saved eval  → {EVAL_V2}")
    print(f"[Retrain] Total time  : {results['elapsed_seconds']}s\n")


if __name__ == "__main__":
    main()
