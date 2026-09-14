"""
01_data_preprocessing.py
SentinelAI: Prompt Injection Firewall for LLM Applications
===========================================================
PURPOSE:
    Load the integrated prompt-injection dataset, inspect it for quality,
    apply the 11-step NLP preprocessing pipeline, and document the
    ML-ready dataset used for all downstream modelling.

INPUT FILES:
    data/processed/SentinelAI_Master_Dataset.csv       (399,741 x 8)
    data/processed/SentinelAI_Preprocessed_Dataset.csv (399,741 x 11)

DATASET SOURCES (documented in project report):
    - Neuralchemy/Prompt-injection-dataset
    - Cyberec/Prompt-injection-dataset
    - hlynlabs/prompt-injection-judge-deberta-dataset

OUTPUT:
    Prints dataset statistics and documents the 27 excluded rows.
    (processed_text already exists in the CSV; this file shows the pipeline
     that generated it — identical to app/preprocess.py for Flask inference.)
"""

import re
import string
import warnings
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

# -- Download NLTK resources -------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# -- Paths (run from project root) -------------------------------------------
BASE         = "."
MASTER_PATH  = f"{BASE}/data/processed/SentinelAI_Master_Dataset.csv"
PREPROC_PATH = f"{BASE}/data/processed/SentinelAI_Preprocessed_Dataset.csv"

# ============================================================================
# SECTION 1 — LOAD THE INTEGRATED DATASET
# ============================================================================
print("=" * 60)
print("SECTION 1 — DATASET LOADING")
print("=" * 60)

# Master dataset: 8 raw columns (text, label, category, source, severity,
#                                group_id, augmented, tags)
master = pd.read_csv(MASTER_PATH, low_memory=False)
print(f"Master dataset shape       : {master.shape}")
print(f"Columns                    : {list(master.columns)}")
print(f"Label counts (0=Benign, 1=Malicious):")
print(master["label"].value_counts().sort_index().to_string())

# Preprocessed dataset: 11 columns (8 + clean_text, tokens, processed_text)
df = pd.read_csv(PREPROC_PATH, low_memory=False)
print(f"\nPreprocessed dataset shape : {df.shape}")
print(f"Columns                    : {list(df.columns)}")

# ============================================================================
# SECTION 2 — NLP PREPROCESSING PIPELINE (CANONICAL)
# ============================================================================
# IMPORTANT: This pipeline is IDENTICAL to app/preprocess.py.
# Any modification here must be reflected in Flask inference as well.
print("\n" + "=" * 60)
print("SECTION 2 — NLP PREPROCESSING PIPELINE")
print("=" * 60)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE    = re.compile(r"<[^>]+>")
_EMAIL_RE   = re.compile(r"\S+@\S+")
_NUMBER_RE  = re.compile(r"\b\d+\b")
_WHITESPACE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    11-step NLP preprocessing pipeline for prompt injection detection.

    Steps:
        1.  Lowercase conversion
        2.  URL removal
        3.  HTML tag removal
        4.  Email address removal
        5.  Number removal
        6.  Punctuation removal
        7.  Whitespace normalisation
        8.  Tokenisation (NLTK word_tokenize)
        9.  Stopword removal (English NLTK stopwords)
        10. Lemmatisation (WordNetLemmatizer)
        11. Token reconstruction into clean string

    Returns "" if input is None or empty after processing.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()                                              # Step 1
    text = _URL_RE.sub(" ", text)                                    # Step 2
    text = _HTML_RE.sub(" ", text)                                   # Step 3
    text = _EMAIL_RE.sub(" ", text)                                  # Step 4
    text = _NUMBER_RE.sub(" ", text)                                 # Step 5
    text = text.translate(str.maketrans("", "", string.punctuation)) # Step 6
    text = _WHITESPACE.sub(" ", text).strip()                        # Step 7
    tokens = word_tokenize(text)                                     # Step 8
    tokens = [
        _LEMMATIZER.lemmatize(tok)                                   # Step 10
        for tok in tokens                                            # Step 9
        if tok.isalpha() and tok not in _STOP_WORDS
    ]
    return " ".join(tokens)                                          # Step 11


# ============================================================================
# SECTION 3 — DATA QUALITY INSPECTION
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 3 — DATA QUALITY INSPECTION")
print("=" * 60)

# Identify rows where processed_text is unusable
problem_mask = df["processed_text"].isna() | \
               (df["processed_text"].fillna("").str.strip() == "")

n_problem   = int(problem_mask.sum())
excl_labels = df[problem_mask]["label"].value_counts().to_dict()

print(f"Total rows                 : {len(df):,}")
print(f"Unusable processed_text    : {n_problem}")
print(f"  - label breakdown        : {excl_labels}")
print(f"\nReason for exclusion:")
print("  Rows that become empty after the NLP pipeline have no TF-IDF features.")
print("  They contain only: HTML tokens (<|im_start|>), pure numbers, single chars,")
print("  or Unicode content stripped by steps 3/4/5/9.")
print(f"\nSample excluded rows:")
for idx, row in df[problem_mask].head(8).iterrows():
    print(f"  idx={idx}  label={row['label']}  raw='{str(row['text'])[:60]}'")

# IMPORTANT: Exclusions happen BEFORE any train/test split.
df_ml = df[~problem_mask].copy()
print(f"\nML-ready rows              : {len(df_ml):,}")
print(f"  Benign    (label=0)      : {(df_ml['label']==0).sum():,}")
print(f"  Malicious (label=1)      : {(df_ml['label']==1).sum():,}")

# ============================================================================
# SECTION 4 — PREPROCESSING CONSISTENCY CROSS-CHECK
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 4 — PREPROCESSING CONSISTENCY CHECK")
print("=" * 60)

# Verify that running the pipeline on raw text reproduces the CSV column.
sample_row = df_ml.iloc[0]
csv_proc   = sample_row["processed_text"]
live_proc  = preprocess_text(sample_row["text"])
match      = csv_proc.strip() == live_proc.strip()

print(f"Sample raw text  : {sample_row['text'][:80]}")
print(f"CSV processed    : '{csv_proc}'")
print(f"Pipeline output  : '{live_proc}'")
print(f"Match            : {'IDENTICAL' if match else 'DIFFERS — INVESTIGATE'}")
