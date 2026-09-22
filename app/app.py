"""
app.py — SentinelAI Flask Web Application  (Dual-Mode Edition)
Prompt Injection Firewall for LLM Applications

Modes accepted via the "mode" POST field:
    fast    — TF-IDF + Logistic Regression (~1-2 ms, stateless)
    deep    — DeBERTa-v3 Transformer (semantic, ~50-150 ms)
    hybrid  — Fast first; escalate to Deep if confidence is ambiguous

Run command:
    cd /home/nakul/Desktop/SentinelAI/app && python app.py
"""
import os
import sys
import time

# ── Resolve paths BEFORE any imports that touch sys.path ──────────────────
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_APP_DIR)

os.chdir(_APP_DIR)

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from flask import Flask, render_template, request, jsonify
import joblib
from preprocess import preprocess_text
import neural_detector

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_BASE_DIR, "models", "logistic_regression.joblib")
VEC_PATH   = os.path.join(_BASE_DIR, "models", "tfidf_vectorizer.joblib")

# Hybrid mode: escalate to Deep if fast-filter confidence falls in this window
HYBRID_AMBIGUITY_LOW  = 20.0   # %
HYBRID_AMBIGUITY_HIGH = 80.0   # %

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ── Load Tier-1 model at startup ──────────────────────────────────────────
print("[SentinelAI] Loading Tier-1 model and vectorizer ...")
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print(f"[SentinelAI] Tier-1 model     : {MODEL_PATH}")
    print(f"[SentinelAI] Tier-1 vectorizer: {VEC_PATH}")
    MODEL_LOADED = True
except Exception as e:
    print(f"[SentinelAI] ERROR loading Tier-1 model: {e}")
    MODEL_LOADED = False

# ── Start loading Tier-2 neural model in background ───────────────────────
neural_detector.load(background=True)


# ── Helper: Tier-1 fast predict ───────────────────────────────────────────
def _fast_predict(prompt_text: str) -> dict:
    """Run TF-IDF + Logistic Regression on the given raw prompt."""
    processed = preprocess_text(prompt_text)

    if not processed:
        return {
            "label": "MALICIOUS", "label_code": 1,
            "confidence": 99.0, "recommendation": "BLOCK",
            "processed_text": "", "latency_ms": 0.0,
            "model_used": "TF-IDF + Logistic Regression",
            "note": "Prompt reduced to empty after preprocessing — likely obfuscated/token-only injection.",
        }

    t0 = time.perf_counter()
    X          = vectorizer.transform([processed])
    pred_code  = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    confidence = round(float(proba[pred_code]) * 100, 2)
    label          = "MALICIOUS" if pred_code == 1 else "SAFE"
    recommendation = "BLOCK"    if pred_code == 1 else "ALLOW"

    return {
        "label": label, "label_code": pred_code,
        "confidence": confidence, "recommendation": recommendation,
        "processed_text": processed, "latency_ms": latency_ms,
        "model_used": "TF-IDF + Logistic Regression",
    }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Tier-1 model not loaded. Check server logs."}), 500

    data        = request.get_json(silent=True) or {}
    prompt_text = data.get("prompt_text", "").strip()
    mode        = data.get("mode", "fast").lower()

    if not prompt_text:
        return jsonify({"error": "No prompt text provided."}), 400

    if mode == "fast":
        result = _fast_predict(prompt_text)
        result["mode"] = "fast"
        return jsonify(result)

    elif mode == "deep":
        nd_status = neural_detector.status()
        if not nd_status["ready"] and nd_status["status"] == "unavailable":
            return jsonify({"error": nd_status["error"]}), 503

        result = neural_detector.predict(prompt_text)
        if "error" in result:
            return jsonify(result), 503

        result["mode"]           = "deep"
        result["processed_text"] = prompt_text  # DeBERTa uses raw text
        return jsonify(result)

    elif mode == "hybrid":
        # Tier 1 — fast filter
        fast_result = _fast_predict(prompt_text)
        conf = fast_result["confidence"]
        is_ambiguous = HYBRID_AMBIGUITY_LOW < conf < HYBRID_AMBIGUITY_HIGH

        if not is_ambiguous:
            # High-confidence fast result — no need for Tier 2
            fast_result["mode"]        = "hybrid"
            fast_result["tier_used"]   = 1
            fast_result["escalated"]   = False
            fast_result["audit_trail"] = (
                f"Tier-1 confidence {conf}% — outside ambiguity window "
                f"({HYBRID_AMBIGUITY_LOW}%–{HYBRID_AMBIGUITY_HIGH}%). "
                f"Decision: {fast_result['recommendation']} (no escalation needed)."
            )
            return jsonify(fast_result)

        # Ambiguous — escalate to Tier 2
        nd_status = neural_detector.status()
        if not nd_status["ready"]:
            # Neural not ready — fall back to fast result with a note
            fast_result["mode"]        = "hybrid"
            fast_result["tier_used"]   = 1
            fast_result["escalated"]   = False
            fast_result["audit_trail"] = (
                f"Tier-1 confidence {conf}% — ambiguous. "
                f"Escalation attempted but Tier-2 model is {nd_status['status']}. "
                f"Falling back to Tier-1 result."
            )
            return jsonify(fast_result)

        deep_result = neural_detector.predict(prompt_text)
        if "error" in deep_result:
            fast_result["mode"]        = "hybrid"
            fast_result["tier_used"]   = 1
            fast_result["escalated"]   = False
            fast_result["audit_trail"] = f"Escalation failed: {deep_result['error']}. Using Tier-1 result."
            return jsonify(fast_result)

        deep_result["mode"]        = "hybrid"
        deep_result["tier_used"]   = 2
        deep_result["escalated"]   = True
        deep_result["tier1_confidence"] = conf
        deep_result["tier1_label"]      = fast_result["label"]
        deep_result["processed_text"]   = fast_result.get("processed_text", "")
        deep_result["audit_trail"] = (
            f"Tier-1 confidence {conf}% — ambiguous window "
            f"({HYBRID_AMBIGUITY_LOW}%–{HYBRID_AMBIGUITY_HIGH}%). "
            f"Escalated to Tier-2 DeBERTa. "
            f"Final decision: {deep_result['recommendation']} "
            f"(confidence {deep_result['confidence']}% in {deep_result['latency_ms']}ms)."
        )
        return jsonify(deep_result)

    else:
        return jsonify({"error": f"Unknown mode '{mode}'. Use fast, deep, or hybrid."}), 400


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "tier1_loaded": MODEL_LOADED,
    })


@app.route("/neural_status")
def neural_status():
    """Endpoint for the UI to poll DeBERTa loading progress."""
    return jsonify(neural_detector.status())


if __name__ == "__main__":
    print(f"[SentinelAI] Working directory: {os.getcwd()}")
    print("[SentinelAI] Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
