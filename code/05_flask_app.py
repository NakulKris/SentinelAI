"""
05_flask_app.py
SentinelAI: Prompt Injection Firewall for LLM Applications
===========================================================
PURPOSE:
    Flask web application that exposes the trained model as a
    Prompt Injection Firewall. Users submit a prompt; the app
    preprocesses it using the same NLP pipeline as training and
    returns a SAFE/MALICIOUS verdict with confidence score.

INTERNAL LABEL MAPPING (model classes, unchanged):
    0 = Benign
    1 = Malicious

UI DISPLAY MAPPING (user-facing only):
    model class 0  ->  displays "SAFE"  /  Action: ALLOW
    model class 1  ->  displays "MALICIOUS"  /  Action: BLOCK

NOTE: The underlying model, dataset labels, and metrics all use
      the original Benign/Malicious terminology. "SAFE" is a
      presentation-layer change in the Flask response only.

RUN COMMAND:
    cd /path/to/SentinelAI/app
    python app.py
    Then open http://127.0.0.1:5000

ROUTES:
    GET  /         -> serves the HTML dashboard (index.html)
    POST /predict  -> accepts JSON {"prompt_text": "..."}, returns verdict
    GET  /health   -> {"status": "ok", "model_loaded": true}
"""

import os
import sys

# -- Resolve paths before imports (avoids NLTK 'regex' path conflict) --------
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_APP_DIR)
os.chdir(_APP_DIR)                          # run from app/ to avoid NLTK issue
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import preprocess_text      # shared NLP pipeline

# -- Model paths -------------------------------------------------------------
MODEL_PATH = os.path.join(_BASE_DIR, "models", "logistic_regression.joblib")
VEC_PATH   = os.path.join(_BASE_DIR, "models", "tfidf_vectorizer.joblib")

app = Flask(__name__)

# -- Load model and vectorizer once at startup --------------------------------
print("[SentinelAI] Loading model and vectorizer ...")
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print(f"[SentinelAI] Model loaded     : {MODEL_PATH}")
    print(f"[SentinelAI] Vectorizer loaded: {VEC_PATH}")
    MODEL_LOADED = True
except Exception as e:
    print(f"[SentinelAI] ERROR loading model: {e}")
    MODEL_LOADED = False


@app.route("/")
def index():
    """Serve the main HTML dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a prompt, run the full inference pipeline, return a verdict.

    Request body (JSON):
        {"prompt_text": "<your prompt here>"}

    Response (JSON):
        {
            "label":          "SAFE" or "MALICIOUS",  <- UI display label
            "label_code":     0 or 1,                 <- model class (unchanged)
            "confidence":     float (0-100),
            "recommendation": "ALLOW" or "BLOCK",
            "processed_text": "<NLP pipeline output>"
        }

    Special case — obfuscated/empty prompt:
        If preprocessing reduces the input to an empty string, the prompt
        is classified as MALICIOUS by a RULE-BASED SAFETY FALLBACK
        (not the ML model). This is documented by a "note" field in the
        response and the confidence is a fixed 99.0 — NOT a model score.
    """
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json(silent=True) or {}
    prompt_text = data.get("prompt_text", "").strip()
    if not prompt_text:
        return jsonify({"error": "No prompt text provided."}), 400

    # Step 1 — NLP preprocessing (identical to training pipeline)
    processed = preprocess_text(prompt_text)

    # Safety fallback: empty processed_text = obfuscated/token-only injection
    if not processed:
        return jsonify({
            "label":          "MALICIOUS",
            "label_code":     1,
            "confidence":     99.0,          # fixed — NOT a model score
            "recommendation": "BLOCK",
            "note": ("Prompt reduced to empty after preprocessing — "
                     "rule-based safety fallback, not ML model prediction."),
            "processed_text": ""
        })

    # Step 2 — TF-IDF transform using the FITTED training vectorizer
    X = vectorizer.transform([processed])

    # Step 3 — Logistic Regression predict
    pred_code  = int(model.predict(X)[0])      # 0=Benign, 1=Malicious
    proba      = model.predict_proba(X)[0]
    confidence = round(float(proba[pred_code]) * 100, 2)

    # Step 4 — Map internal class to UI display label
    # Internal class labels (0/1) remain unchanged for academic correctness.
    # Only the JSON "label" field uses the display-friendly term "SAFE".
    label          = "MALICIOUS" if pred_code == 1 else "SAFE"
    recommendation = "BLOCK"    if pred_code == 1 else "ALLOW"

    return jsonify({
        "label":          label,
        "label_code":     pred_code,     # 0 or 1 — internal model class
        "confidence":     confidence,
        "recommendation": recommendation,
        "processed_text": processed
    })


@app.route("/health")
def health():
    """Health check endpoint — confirms model is loaded."""
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})


if __name__ == "__main__":
    print(f"[SentinelAI] Working directory: {os.getcwd()}")
    print("[SentinelAI] Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
