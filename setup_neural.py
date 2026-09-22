"""
setup_neural.py — SentinelAI One-Time Neural Model Setup
=========================================================
Downloads ProtectAI/deberta-v3-base-prompt-injection-v2 from HuggingFace
and saves it permanently to models/neural/ on your local disk.

Run ONCE from the project root:
    python setup_neural.py

After this, the Flask app loads the model from disk — no internet needed.
The models/neural/ folder is in .gitignore (too large for GitHub ~750 MB).
"""

import os
import sys

MODEL_ID  = "ProtectAI/deberta-v3-base-prompt-injection-v2"
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "neural")
WEIGHTS   = os.path.join(SAVE_PATH, "model.safetensors")


def main():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[setup] ERROR: transformers not installed.")
        print("        Run:  pip install torch transformers")
        sys.exit(1)

    # Check weights file specifically — config.json alone is not enough
    if os.path.exists(WEIGHTS):
        size_mb = os.path.getsize(WEIGHTS) / (1024 * 1024)
        print(f"[setup] Model weights already saved ({size_mb:.0f} MB) at: {SAVE_PATH}")
        print("[setup] Nothing to do. Ready to use.")
        return

    if os.path.exists(SAVE_PATH):
        print(f"[setup] Partial download detected in {SAVE_PATH} — resuming/retrying...")
    else:
        os.makedirs(SAVE_PATH, exist_ok=True)

    print(f"[setup] Downloading model : {MODEL_ID}")
    print(f"[setup] Save location     : {SAVE_PATH}")
    print("[setup] This may take a few minutes (~750 MB)...")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(SAVE_PATH)
    print("[setup] Tokenizer saved.")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.save_pretrained(SAVE_PATH)
    print()
    print(f"[setup] Model weights saved to: {SAVE_PATH}")
    print("[setup] Setup complete! Flask app will now load from disk — no internet needed.")


if __name__ == "__main__":
    main()
