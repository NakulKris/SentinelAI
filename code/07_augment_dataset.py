"""
07_augment_dataset.py — SentinelAI Dataset Augmentation v2
===========================================================
Downloads real-world adversarial datasets from HuggingFace and merges
them into the existing SentinelAI training data.

Verified working datasets:
    1. deepset/prompt-injections          — gold standard, text/label cols
    2. rubend18/ChatGPT-Jailbreak-Prompts — DAN/AIM/roleplay, Prompt col
    3. JasperLS/prompt-injections         — text/label cols (extended version)

Run from project root:
    python code/07_augment_dataset.py
"""

import os, sys, json, time
import pandas as pd

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MASTER_V1     = os.path.join(PROCESSED_DIR, "SentinelAI_Master_Dataset.csv")
MASTER_V2     = os.path.join(PROCESSED_DIR, "SentinelAI_Master_Dataset_v2.csv")
REPORT_PATH   = os.path.join(PROCESSED_DIR, "augmentation_report.json")


def fetch(ds_id, split, text_col, label_col=None, label_map=None, fixed_label=None):
    """Download one HuggingFace dataset and return a clean DataFrame."""
    from datasets import load_dataset
    print(f"  ↓ {ds_id} ...")
    try:
        ds  = load_dataset(ds_id, split=split)
        df  = ds.to_pandas()

        # Resolve text column
        if text_col not in df.columns:
            for alt in ["prompt", "Prompt", "input", "content", "question", "instruction"]:
                if alt in df.columns:
                    text_col = alt
                    break

        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str).str.strip()

        if label_col and label_col in df.columns:
            result["label"] = df[label_col].map(label_map).astype(int)
        else:
            result["label"] = int(fixed_label)

        result["source"] = ds_id
        result = result[result["text"].str.len() > 15].copy()

        mal = int((result["label"]==1).sum())
        ben = int((result["label"]==0).sum())
        print(f"     {len(result):,} rows  |  Malicious: {mal:,}  Benign: {ben:,}")
        return result

    except Exception as e:
        print(f"     ✗ {e}")
        return pd.DataFrame(columns=["text","label","source"])


def main():
    t0 = time.perf_counter()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    # ── Load existing master ──────────────────────────────────────────────
    print("\n[Augment] Loading existing master dataset …")
    df_orig = pd.read_csv(MASTER_V1, usecols=["text","label"], low_memory=False)
    df_orig["text"]   = df_orig["text"].astype(str).str.strip()
    df_orig["label"]  = df_orig["label"].astype(int)
    df_orig["source"] = "SentinelAI_v1"
    print(f"  Existing : {len(df_orig):,} rows  |  Mal: {(df_orig['label']==1).sum():,}  Ben: {(df_orig['label']==0).sum():,}")

    seen = set(df_orig["text"].str.lower().str.strip())

    # ── HuggingFace sources ───────────────────────────────────────────────
    SOURCES = [
        # text + binary label (1=injection, 0=legit)
        dict(ds_id="deepset/prompt-injections",          split="train", text_col="text",   label_col="label", label_map={1:1, 0:0}),
        dict(ds_id="JasperLS/prompt-injections",         split="train", text_col="text",   label_col="label", label_map={1:1, 0:0}),
        # all jailbreaks → label 1, correct column is "Prompt"
        dict(ds_id="rubend18/ChatGPT-Jailbreak-Prompts", split="train", text_col="Prompt", fixed_label=1),
    ]

    print("\n[Augment] Downloading external datasets …")
    new_frames, stats = [], {}

    for cfg in SOURCES:
        df_new = fetch(**cfg)
        if df_new.empty:
            stats[cfg["ds_id"]] = {"added": 0}
            continue

        before   = len(df_new)
        df_new   = df_new[~df_new["text"].str.lower().str.strip().isin(seen)].copy()
        dupes    = before - len(df_new)
        seen.update(df_new["text"].str.lower().str.strip())

        new_frames.append(df_new)
        stats[cfg["ds_id"]] = {
            "downloaded": before, "dupes_removed": dupes,
            "added": len(df_new),
            "malicious": int((df_new["label"]==1).sum()),
            "benign":    int((df_new["label"]==0).sum()),
        }

    # ── Combine ───────────────────────────────────────────────────────────
    if new_frames:
        df_all = pd.concat([df_orig] + new_frames, ignore_index=True)
    else:
        df_all = df_orig

    print(f"\n[Augment] Combined: {len(df_all):,} rows  |  "
          f"Mal: {(df_all['label']==1).sum():,}  Ben: {(df_all['label']==0).sum():,}")

    df_all[["text","label","source"]].to_csv(MASTER_V2, index=False)
    print(f"[Augment] Saved  → {MASTER_V2}")

    elapsed = round(time.perf_counter() - t0, 1)
    report = {
        "original_rows": len(df_orig),
        "combined_rows": len(df_all),
        "new_rows_added": len(df_all) - len(df_orig),
        "sources": stats,
        "elapsed_seconds": elapsed,
    }
    with open(REPORT_PATH,"w") as f:
        json.dump(report, f, indent=2)

    print(f"[Augment] Report → {REPORT_PATH}")
    print(f"[Augment] Done in {elapsed}s\n")
    print("Next step: python code/08_retrain_augmented.py")


if __name__ == "__main__":
    main()
