# SentinelAI — Academic Code Package

**Project:** SentinelAI: Prompt Injection Firewall for LLM Applications    
**Model:** TF-IDF (50,000 features) + Logistic Regression  
**Accuracy:** 94.40% | F1: 94.25% | ROC-AUC: 98.67%

---

## File Descriptions

| File | Purpose | Source |
|---|---|---|
| `01_data_preprocessing.py` | Dataset loading, NLP pipeline definition, quality inspection | Derived from `app/preprocess.py` + execution scripts |
| `02_eda.py` | Generate all 5 EDA figures (label, category, severity, char-length, word-count) | Derived from `run_ml_pipeline.py` Steps 3–4 |
| `03_train_model.py` | Leakage-free train/test split, TF-IDF fit, LR training, save artifacts | Derived from `retrain_leakage_free.py` (final authoritative training run) |
| `04_evaluate_model.py` | Reload saved model, independently recalculate and verify metrics | Derived from `sentinelai_qa.py` Check 6 |
| `05_flask_app.py` | Flask web application — inference pipeline and web routes | Derived from `app/app.py` (with documentation expanded) |
| `ppt_code_snippets.txt` | Short slide-ready code blocks for presentation | New — condensed from above files |

---

## Execution Order

Run scripts in this order from the **project root** (`/home/nakul/Desktop/SentinelAI/`):

```bash
# Step 1 — Inspect dataset and verify preprocessing
python code/01_data_preprocessing.py

# Step 2 — Generate EDA figures (saved to figures/)
python code/02_eda.py

# Step 3 — Train the model (saves to models/)
#          WARNING: This takes ~3 minutes (400k rows x 50k features)
python code/03_train_model.py

# Step 4 — Verify saved model metrics independently
python code/04_evaluate_model.py

# Step 5 — Run the Flask application
#          MUST be run from app/ directory
cd app && python app.py
# Then open http://127.0.0.1:5000
```

---

## Required Input Files

| File | Size | SHA256 |
|---|---|---|
| `data/processed/SentinelAI_Master_Dataset.csv` | 204.8 MB | `9b96e1b...` |
| `data/processed/SentinelAI_Preprocessed_Dataset.csv` | 715 MB | `ac5970...` |

---

## Generated Outputs

### EDA Figures (`figures/`)
- `01_label_distribution.png` — Benign vs Malicious bar chart
- `02_category_distribution.png` — Attack category breakdown (Neuralchemy subset)
- `03_severity_distribution.png` — Severity levels (Neuralchemy subset)
- `04_char_length_distribution.png` — Prompt character-length histogram
- `05_word_count_distribution.png` — Prompt word-count histogram

### Model Artifacts (`models/`)
- `logistic_regression.joblib` — Trained Logistic Regression model
- `tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer (50,000 features)
- `evaluation_results.json` — All metrics in JSON format

### Model Figures (`figures/`)
- `06_confusion_matrix.png` — Confusion matrix plot

---

## Dataset Facts

| Field | Value |
|---|---|
| Full preprocessed dataset rows | 399,741 |
| Full dataset Benign (label=0) | 203,067 |
| Full dataset Malicious (label=1) | 196,674 |
| Excluded (unusable processed_text) | 27 (14 Malicious, 13 Benign) |
| **Final ML-ready rows** | **399,714** |
| **ML-ready Benign (label=0)** | **203,054** |
| **ML-ready Malicious (label=1)** | **196,660** |
| Text feature | `processed_text` |
| Target variable | `label` |

> **Note:** The Benign/Malicious class counts used for ML training and evaluation are the **ML-ready** figures (203,054 / 196,660), not the full dataset figures (203,067 / 196,674). The EDA label distribution figure uses the ML-ready set. The EDA category and severity figures represent only the Neuralchemy subset (~4,391 rows with named categories / ~2,640 rows with severity labels) — not the full 399,714-row corpus.

---

## Model Configuration

### Train/Test Split
- **Method:** Grouped approximately 80/20 train-test split by unique `processed_text`, `random_state=42`
- **No `stratify=` argument** — class balance is preserved approximately by the large sample size (397,427 unique texts)
- **Result:** 319,745 train rows / 79,969 test rows
- **Leakage check:** 0 overlapping `processed_text` values between partitions

### TF-IDF
```python
TfidfVectorizer(
    max_features  = 50000,
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
    max_df        = 0.95,
    strip_accents = "unicode",
    analyzer      = "word",
    token_pattern = r"\b[a-zA-Z]\w+\b"
)
```

### Logistic Regression
```python
LogisticRegression(
    C            = 1.0,
    max_iter     = 1000,
    solver       = "saga",
    random_state = 42,
    class_weight = None,
    n_jobs       = -1
)
```

---

## Final Metrics (Leakage-Free)

| Metric | Value |
|---|---|
| Accuracy | **94.40%** |
| Precision | **95.01%** |
| Recall | **93.51%** |
| F1-score | **94.25%** |
| ROC-AUC | **98.67%** |
| TN | 38,770 |
| FP | 1,928 |
| FN | 2,550 |
| TP | 36,721 |

---

## Technical Note: UI Terminology

The Flask application (`05_flask_app.py`) displays **"SAFE"** in the web
interface when the model predicts class 0 (Benign). This is a
**presentation-layer change only**. The model, dataset labels, and all
evaluation metrics use the original `0 = Benign / 1 = Malicious` encoding.
