"""
app.py — SentinelAI Flask Web Application
Prompt Injection Firewall for LLM Applications

Run command:
    cd /home/nakul/Desktop/SentinelAI/app && python app.py
    OR:
    python -P /home/nakul/Desktop/SentinelAI/app/app.py
"""
import os
import sys

# ── Resolve paths BEFORE any imports that touch sys.path ─────────────────
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_APP_DIR)

# Change CWD to app/ so NLTK's security guard doesn't flag the project root
os.chdir(_APP_DIR)

# Ensure app/ is on sys.path for preprocess import
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import preprocess_text

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_BASE_DIR, "models", "logistic_regression.joblib")
VEC_PATH   = os.path.join(_BASE_DIR, "models", "tfidf_vectorizer.joblib")

app = Flask(__name__)

# ── Load model and vectorizer at startup ──────────────────────────────────
print("[SentinelAI] Loading model and vectorizer …")
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
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.get_json(silent=True) or {}
    prompt_text = data.get("prompt_text", "").strip()

    if not prompt_text:
        return jsonify({"error": "No prompt text provided."}), 400

    # Step 1: NLP preprocessing (identical pipeline to training)
    processed = preprocess_text(prompt_text)

    if not processed:
        return jsonify({
            "label":          "MALICIOUS",
            "label_code":     1,
            "confidence":     99.0,
            "recommendation": "BLOCK",
            "note":           "Prompt reduced to empty after preprocessing — likely obfuscated/token-only injection.",
            "processed_text": ""
        })

    # Step 2: TF-IDF transform using the FITTED training vectorizer
    X = vectorizer.transform([processed])

    # Step 3: Logistic Regression predict
    pred_code  = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    confidence = round(float(proba[pred_code]) * 100, 2)

    label          = "MALICIOUS" if pred_code == 1 else "SAFE"
    recommendation = "BLOCK"    if pred_code == 1 else "ALLOW"

    return jsonify({
        "label":          label,
        "label_code":     pred_code,
        "confidence":     confidence,
        "recommendation": recommendation,
        "processed_text": processed
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})


if __name__ == "__main__":
    print(f"[SentinelAI] Working directory: {os.getcwd()}")
    print("[SentinelAI] Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
