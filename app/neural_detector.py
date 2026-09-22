"""
neural_detector.py — SentinelAI Tier-2 Deep Semantic Detector

Load order (fastest to slowest):
    1. Local disk: ../models/neural/   ← instant, no internet
    2. HuggingFace Hub (fallback)      ← requires internet, ~500 MB download

Label mapping:
    HuggingFace label "INJECTION"   -> label_code=1, label="MALICIOUS"
    HuggingFace label "LEGITIMATE"  -> label_code=0, label="SAFE"

Run setup_neural.py once to download and save the model locally.
"""

import os
import time
import threading

# ── Paths ──────────────────────────────────────────────────────────────────
_APP_DIR      = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR     = os.path.dirname(_APP_DIR)
LOCAL_MODEL   = os.path.join(_BASE_DIR, "models", "neural")
REMOTE_MODEL  = "ProtectAI/deberta-v3-base-prompt-injection-v2"

# Use local path if it exists and has the config, else fall back to HuggingFace
MODEL_SOURCE  = LOCAL_MODEL if os.path.exists(os.path.join(LOCAL_MODEL, "config.json")) else REMOTE_MODEL

LABEL_MAP = {
    "INJECTION":  (1, "MALICIOUS", "BLOCK"),
    "LEGITIMATE": (0, "SAFE",      "ALLOW"),
}

# ── Module-level state ─────────────────────────────────────────────────────
_pipeline  = None
_load_lock = threading.Lock()
_status    = "not_loaded"   # not_loaded | loading | ready | unavailable
_error_msg = ""


# ── Public API ─────────────────────────────────────────────────────────────

def status() -> dict:
    return {
        "status":       _status,
        "model_source": "local disk" if MODEL_SOURCE == LOCAL_MODEL else "HuggingFace Hub",
        "model":        REMOTE_MODEL,
        "error":        _error_msg,
        "ready":        _status == "ready",
    }


def load(background: bool = False) -> bool:
    """
    Load the DeBERTa pipeline.
    Set background=True to load in a daemon thread (non-blocking).
    """
    global _status
    if _status in ("ready", "loading"):
        return True
    if _status == "unavailable":
        return False
    if background:
        t = threading.Thread(target=_do_load, daemon=True)
        t.start()
        return True
    return _do_load()


def predict(text: str) -> dict:
    """
    Run deep semantic classification on raw (un-preprocessed) text.
    DeBERTa works best on the original text, NOT on stopword-stripped tokens.
    """
    global _status
    if _status == "not_loaded":
        _do_load()
    if _status == "unavailable":
        return {"error": f"Neural model unavailable. {_error_msg}", "model_used": REMOTE_MODEL}
    if _status == "loading":
        return {"error": "Neural model is still loading. Please retry in a few seconds.", "model_used": REMOTE_MODEL}

    if not text or not text.strip():
        return {
            "label": "MALICIOUS", "label_code": 1, "confidence": 99.0,
            "recommendation": "BLOCK", "latency_ms": 0.0,
            "model_used": "DeBERTa-v3 Semantic (ProtectAI)",
            "note": "Empty input — likely obfuscated/token-only injection.",
        }

    t0 = time.perf_counter()
    result = _pipeline(text)[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    hf_label = result["label"].upper()
    score    = float(result["score"])
    hf_label = "INJECTION" if "INJECT" in hf_label else "LEGITIMATE"

    label_code, label, recommendation = LABEL_MAP[hf_label]
    confidence = round(score * 100, 2)

    return {
        "label": label, "label_code": label_code, "confidence": confidence,
        "recommendation": recommendation, "latency_ms": latency_ms,
        "model_used": "DeBERTa-v3 Semantic (ProtectAI)",
    }


# ── Private ────────────────────────────────────────────────────────────────

def _do_load() -> bool:
    global _pipeline, _status, _error_msg
    with _load_lock:
        if _status == "ready":
            return True
        _status = "loading"
        try:
            from transformers import pipeline as hf_pipeline

            source_label = "local disk" if MODEL_SOURCE == LOCAL_MODEL else "HuggingFace Hub"
            print(f"[SentinelAI] Loading neural model from {source_label}: {MODEL_SOURCE}")
            t0 = time.perf_counter()

            _pipeline = hf_pipeline(
                "text-classification",
                model=MODEL_SOURCE,
                truncation=True,
                max_length=512,
            )
            elapsed = time.perf_counter() - t0
            print(f"[SentinelAI] Neural model ready in {elapsed:.1f}s  (source: {source_label})")
            _status = "ready"
            return True

        except ImportError:
            _error_msg = (
                "torch/transformers not installed. "
                "Run: pip install torch transformers"
            )
            _status = "unavailable"
            print(f"[SentinelAI] Neural mode unavailable: {_error_msg}")
            return False
        except Exception as exc:
            _error_msg = str(exc)
            _status = "unavailable"
            print(f"[SentinelAI] Neural model failed to load: {exc}")
            return False
