# SentinelAI — Prompt Injection Firewall

A dual-mode prompt injection detection firewall for LLM applications.

- **Fast Filter** (⚡): TF-IDF + Logistic Regression · ~1 ms · no GPU needed
- **Deep Semantic** (🧠): DeBERTa-v3 Transformer (ProtectAI) · ~80 ms · detects paraphrased & roleplay attacks
- **Hybrid Cascade** (🛡️): Auto-escalates ambiguous queries from Fast → Deep

**Model performance (Fast Filter):** Accuracy 94.40% | F1 94.25% | ROC-AUC 98.67%

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<YOUR_USERNAME>/SentinelAI.git
cd SentinelAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. (One-time) Download the DeBERTa neural model to disk (~500 MB)
#    Only needed for Deep Semantic / Hybrid Cascade modes
python setup_neural.py

# 4. Run the web app
cd app && python app.py
# Open http://127.0.0.1:5000
```

> **Note:** Step 3 only needs to be run once. The model is saved to `models/neural/`
> and loaded from disk on every subsequent run — no internet required after setup.

---

## Project Structure

```
SentinelAI/
├── app/
│   ├── app.py              # Flask app (Fast / Deep / Hybrid modes)
│   ├── preprocess.py       # NLP preprocessing pipeline (shared with training)
│   ├── neural_detector.py  # DeBERTa-v3 wrapper with local-first loading
│   ├── templates/          # HTML UI
│   └── static/             # CSS
├── code/                   # Academic pipeline scripts (01–06)
├── models/
│   ├── logistic_regression.joblib  # Trained Tier-1 model
│   ├── tfidf_vectorizer.joblib     # Fitted TF-IDF vectorizer
│   ├── evaluation_results.json     # Metrics
│   └── neural/                     # DeBERTa weights (gitignored, run setup_neural.py)
├── figures/                # EDA + evaluation charts
├── data/                   # Datasets (gitignored — too large)
├── setup_neural.py         # One-time neural model download script
└── requirements.txt
```

---

## Tier-1 Model Metrics (Leakage-Free)

| Metric | Value |
|---|---|
| Accuracy | 94.40% |
| Precision | 95.01% |
| Recall | 93.51% |
| F1-Score | 94.25% |
| ROC-AUC | 98.67% |

---

## Run Adversarial Benchmark

```bash
python code/06_benchmark_comparison.py
```

Compares Tier-1 vs Tier-2 on 20 adversarial prompts: direct injections, DAN jailbreaks,
paraphrased attacks, roleplay, zero-width character obfuscation, and base64 hints.
