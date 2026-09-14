"""
03_train_model.py
SentinelAI: Prompt Injection Firewall for LLM Applications
===========================================================
PURPOSE:
    Train the final leakage-free TF-IDF + Logistic Regression model.

    Uses a GROUPED train/test split: unique processed_text values are split
    approximately 80/20 (random_state=42, no stratify= argument), and each
    group is then mapped back to its rows. This prevents any NLP-normalised
    text from appearing in both train AND test partitions.

INPUT FILES:
    data/processed/SentinelAI_Preprocessed_Dataset.csv

OUTPUT FILES:
    models/logistic_regression.joblib   (trained model)
    models/tfidf_vectorizer.joblib      (fitted TF-IDF vectorizer)
    models/evaluation_results.json      (all metrics)
    figures/06_confusion_matrix.png     (confusion matrix plot)

FINAL VERIFIED METRICS (leakage-free):
    Accuracy  : 94.40%
    Precision : 95.01%
    Recall    : 93.51%
    F1-score  : 94.25%
    ROC-AUC   : 98.67%
    TN=38,770  FP=1,928  FN=2,550  TP=36,721
"""

import os
import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings("ignore")

# -- Paths (run from project root) -------------------------------------------
BASE         = "."
PREPROC_PATH = f"{BASE}/data/processed/SentinelAI_Preprocessed_Dataset.csv"
MODEL_PATH   = f"{BASE}/models/logistic_regression.joblib"
VEC_PATH     = f"{BASE}/models/tfidf_vectorizer.joblib"
EVAL_PATH    = f"{BASE}/models/evaluation_results.json"
FIG_PATH     = f"{BASE}/figures/06_confusion_matrix.png"

os.makedirs(f"{BASE}/models",  exist_ok=True)
os.makedirs(f"{BASE}/figures", exist_ok=True)

# ============================================================================
# STEP 1 — LOAD AND PREPARE ML-READY DATASET
# ============================================================================
print("=" * 60)
print("STEP 1 — LOAD DATASET")
print("=" * 60)

df = pd.read_csv(PREPROC_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Exclude rows where processed_text is empty (27 rows — see 01_data_preprocessing.py)
problem_mask = df["processed_text"].isna() | \
               (df["processed_text"].fillna("").str.strip() == "")
df_ml = df[~problem_mask].copy()
print(f"ML-ready rows: {len(df_ml):,}  (excluded: {problem_mask.sum()})")
print(f"  Benign (0)   : {(df_ml['label']==0).sum():,}")
print(f"  Malicious (1): {(df_ml['label']==1).sum():,}")

# ============================================================================
# STEP 2 — LEAKAGE-FREE GROUPED TRAIN/TEST SPLIT
# ============================================================================
# PROBLEM with a naive random split: NLP normalisation collapses different raw
# prompts into identical processed_text values (e.g. "Hello." and "Hello!").
# A random split places some of these collisions in BOTH train and test.
#
# FIX: Split on UNIQUE processed_text values, then map back to rows.
# This guarantees zero overlap of normalised text between partitions.
#
# NOTE: No stratify= argument is used. The near-50/50 class balance is
# preserved approximately by the large sample size (397,427 unique texts).
print("\n" + "=" * 60)
print("STEP 2 — LEAKAGE-FREE GROUPED SPLIT (random_state=42, no stratify)")
print("=" * 60)

RANDOM_STATE = 42

unique_texts = df_ml["processed_text"].astype(str).unique()
print(f"Unique processed_text values: {len(unique_texts):,}")

# Split unique texts approximately 80/20 (no stratify= argument)
train_texts, test_texts = train_test_split(
    list(unique_texts), test_size=0.20, random_state=RANDOM_STATE
)
train_set = set(train_texts)
test_set  = set(test_texts)

# Map back to full dataframe rows
df_train = df_ml[df_ml["processed_text"].astype(str).isin(train_set)]
df_test  = df_ml[df_ml["processed_text"].astype(str).isin(test_set)]

X_train = df_train["processed_text"].astype(str)
y_train = df_train["label"].astype(int)
X_test  = df_test["processed_text"].astype(str)
y_test  = df_test["label"].astype(int)

train_pct = len(df_train) / len(df_ml) * 100
test_pct  = len(df_test)  / len(df_ml) * 100

print(f"Train: {len(df_train):,} rows ({train_pct:.2f}%)")
print(f"Test : {len(df_test):,} rows ({test_pct:.2f}%)")
print(f"Train class dist: {dict(y_train.value_counts().sort_index())}")
print(f"Test  class dist: {dict(y_test.value_counts().sort_index())}")

# VERIFY: zero overlap
overlap = set(X_train) & set(X_test)
assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} overlapping texts!"
print(f"\nLeakage check: PASS — 0 overlapping processed_text values")

# ============================================================================
# STEP 3 — TF-IDF FEATURE EXTRACTION (fit ONLY on training data)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3 — TF-IDF VECTORISATION")
print("=" * 60)

TFIDF_PARAMS = dict(
    max_features  = 50000,       # vocabulary capped at 50,000 terms
    ngram_range   = (1, 2),      # unigrams + bigrams
    sublinear_tf  = True,        # apply log(1 + tf) to dampen high freq
    min_df        = 2,           # ignore terms in fewer than 2 documents
    max_df        = 0.95,        # ignore terms in >95% of documents
    strip_accents = "unicode",
    analyzer      = "word",
    token_pattern = r"\b[a-zA-Z]\w+\b"
)

print("TF-IDF parameters:")
for k, v in TFIDF_PARAMS.items():
    print(f"  {k:20s}: {v}")

# FIT only on training data — test data is ONLY transformed, never fitted
vectorizer    = TfidfVectorizer(**TFIDF_PARAMS)
X_train_tfidf = vectorizer.fit_transform(X_train)   # fit + transform
X_test_tfidf  = vectorizer.transform(X_test)          # transform only

n_features = X_train_tfidf.shape[1]
print(f"\nVocabulary size (TF-IDF features): {n_features:,}")
print(f"Train matrix: {X_train_tfidf.shape}")
print(f"Test  matrix: {X_test_tfidf.shape}")

# ============================================================================
# STEP 4 — LOGISTIC REGRESSION TRAINING
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4 — LOGISTIC REGRESSION TRAINING")
print("=" * 60)

LR_PARAMS = dict(
    C            = 1.0,       # inverse regularisation strength
    max_iter     = 1000,
    solver       = "saga",    # efficient for large sparse matrices
    random_state = RANDOM_STATE,
    class_weight = None,      # dataset is nearly balanced — no weighting
    n_jobs       = -1         # use all CPU cores
)

print("Logistic Regression parameters:")
for k, v in LR_PARAMS.items():
    print(f"  {k:20s}: {v}")

print("\nTraining ...")
model = LogisticRegression(**LR_PARAMS)
model.fit(X_train_tfidf, y_train)
print("Training complete.")
print(f"  Classes: {model.classes_}  (0=Benign, 1=Malicious)")
print(f"  Features: {model.n_features_in_:,}")

# ============================================================================
# STEP 5 — MODEL EVALUATION (test set only)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5 — EVALUATION ON TEST SET")
print("=" * 60)

y_pred  = model.predict(X_test_tfidf)
y_prob  = model.predict_proba(X_test_tfidf)[:, 1]  # P(Malicious)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="binary")
rec  = recall_score(y_test, y_pred, average="binary")
f1   = f1_score(y_test, y_pred, average="binary")
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"Accuracy  : {acc:.6f}  ({acc*100:.2f}%)")
print(f"Precision : {prec:.6f}  ({prec*100:.2f}%)")
print(f"Recall    : {rec:.6f}  ({rec*100:.2f}%)")
print(f"F1-score  : {f1:.6f}  ({f1*100:.2f}%)")
print(f"ROC-AUC   : {auc:.6f}  ({auc*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  TN={TN:,}  FP={FP:,}")
print(f"  FN={FN:,}   TP={TP:,}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

# ============================================================================
# STEP 6 — SAVE ARTIFACTS
# ============================================================================
print("=" * 60)
print("STEP 6 — SAVING ARTIFACTS")
print("=" * 60)

# Save trained model and fitted vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VEC_PATH)
print(f"Model saved      : {MODEL_PATH}")
print(f"Vectorizer saved : {VEC_PATH}")

# Save evaluation metrics to JSON
eval_results = {
    "model": "Logistic Regression (Leakage-Free)",
    "vectorizer": "TF-IDF",
    "tfidf_params": {k: str(v) for k, v in TFIDF_PARAMS.items()},
    "n_tfidf_features": int(n_features),
    "train_rows": int(len(df_train)),
    "test_rows": int(len(df_test)),
    "train_test_split": f"{train_pct:.1f}/{test_pct:.1f} (Grouped by processed_text)",
    "random_state": RANDOM_STATE,
    "leakage_free": True,
    "accuracy":  round(float(acc), 6),
    "precision": round(float(prec), 6),
    "recall":    round(float(rec), 6),
    "f1_score":  round(float(f1), 6),
    "roc_auc":   round(float(auc), 6),
    "confusion_matrix": cm.tolist()
}
with open(EVAL_PATH, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"Metrics saved    : {EVAL_PATH}")

# Save confusion matrix figure
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=["Benign", "Malicious"],
       yticklabels=["Benign", "Malicious"],
       xlabel="Predicted Label", ylabel="True Label",
       title="Confusion Matrix — Leakage-Free Logistic Regression")
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved     : {FIG_PATH}")
