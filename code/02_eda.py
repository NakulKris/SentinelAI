"""
02_eda.py
SentinelAI: Prompt Injection Firewall for LLM Applications
===========================================================
PURPOSE:
    Exploratory Data Analysis on the ML-ready dataset.
    Generates all five EDA figures used in the project presentation.

INPUT FILES:
    data/processed/SentinelAI_Preprocessed_Dataset.csv

OUTPUT FILES:
    figures/01_label_distribution.png
    figures/02_category_distribution.png
    figures/03_severity_distribution.png
    figures/04_char_length_distribution.png
    figures/05_word_count_distribution.png
"""

import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# -- Paths (run from project root) -------------------------------------------
BASE         = "."
PREPROC_PATH = f"{BASE}/data/processed/SentinelAI_Preprocessed_Dataset.csv"
FIGURES_DIR  = f"{BASE}/figures"

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"Benign": "#2ecc71", "Malicious": "#e74c3c"}

# ============================================================================
# LOAD DATASET AND PREPARE ML-READY SUBSET
# ============================================================================
print("Loading preprocessed dataset ...")
df = pd.read_csv(PREPROC_PATH, low_memory=False)
print(f"  Shape: {df.shape}")

# Exclude 27 rows with unusable processed_text (same exclusion as ML pipeline)
problem_mask = df["processed_text"].isna() | \
               (df["processed_text"].fillna("").str.strip() == "")
df_ml = df[~problem_mask].copy()
print(f"  ML-ready rows: {len(df_ml):,}")

# Compute text statistics on the raw 'text' column
df_ml["char_len"]   = df_ml["text"].str.len()
df_ml["word_count"] = df_ml["text"].str.split().str.len()

# ============================================================================
# FIGURE 1 — LABEL DISTRIBUTION (Benign vs Malicious)
# ============================================================================
counts = df_ml["label"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Benign (0)", "Malicious (1)"],
    [counts.get(0, 0), counts.get(1, 0)],
    color=["#2ecc71", "#e74c3c"],
    edgecolor="black", linewidth=0.8, width=0.5
)
for bar, cnt in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1500,
            f"{cnt:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_title("Label Distribution — SentinelAI Dataset", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Number of Prompts", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(counts.values) * 1.15)
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/01_label_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig 1 saved | Benign: {counts.get(0,0):,}  Malicious: {counts.get(1,0):,}")

# ============================================================================
# FIGURE 2 — ATTACK CATEGORY DISTRIBUTION (Neuralchemy subset only)
# ============================================================================
# Only 4,391 rows (Neuralchemy) have named categories; Hlyn rows = "unknown"
cat_meaningful = df_ml[df_ml["category"] != "unknown"]["category"]
cat_counts = cat_meaningful.value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if cat != "benign" else "#2ecc71" for cat in cat_counts.index]
bars2 = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
                color=colors[::-1], edgecolor="black", linewidth=0.5)
for bar, val in zip(bars2, cat_counts.values[::-1]):
    ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)
ax.set_title("Attack Category Distribution\n(Neuralchemy subset — 4,391 labelled rows)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Count", fontsize=11)
ax.set_ylabel("Category", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/02_category_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig 2 saved | Named categories: {len(cat_meaningful):,} rows")

# ============================================================================
# FIGURE 3 — SEVERITY DISTRIBUTION (Neuralchemy subset only)
# ============================================================================
sev_meaningful = df_ml[df_ml["severity"].notna() &
                        (df_ml["severity"].str.strip() != "")]["severity"]
sev_counts = sev_meaningful.value_counts()

sev_order  = ["critical", "high", "medium", "low"]
sev_colors = {"critical": "#c0392b", "high": "#e74c3c",
              "medium": "#f39c12", "low": "#2ecc71"}
sev_vals   = [sev_counts.get(s, 0) for s in sev_order]
sev_cols   = [sev_colors[s] for s in sev_order]

fig, ax = plt.subplots(figsize=(7, 5))
bars3 = ax.bar(sev_order, sev_vals, color=sev_cols,
               edgecolor="black", linewidth=0.8, width=0.5)
for bar, val in zip(bars3, sev_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Severity Distribution\n(Neuralchemy subset — rows with severity labels)",
             fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Severity Level", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/03_severity_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Fig 3 saved | {dict(zip(sev_order, sev_vals))}")

# ============================================================================
# FIGURE 4 — CHARACTER LENGTH DISTRIBUTION
# ============================================================================
char_cap = int(df_ml["char_len"].quantile(0.99))
fig, ax = plt.subplots(figsize=(9, 5))
for label_val, color, name in [(0, "#2ecc71", "Benign"), (1, "#e74c3c", "Malicious")]:
    subset = df_ml[df_ml["label"] == label_val]["char_len"]
    ax.hist(subset, bins=80, alpha=0.6, color=color, label=name,
            range=(0, min(2000, char_cap)))
ax.set_title("Prompt Character Length Distribution", fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Number of Characters", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/04_char_length_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
char_stats = df_ml.groupby("label")["char_len"].agg(["mean", "median", "max"]).round(1)
print(f"Fig 4 saved | Char length stats:\n{char_stats.to_string()}")

# ============================================================================
# FIGURE 5 — WORD COUNT DISTRIBUTION
# ============================================================================
word_cap = int(df_ml["word_count"].quantile(0.99))
fig, ax = plt.subplots(figsize=(9, 5))
for label_val, color, name in [(0, "#2ecc71", "Benign"), (1, "#e74c3c", "Malicious")]:
    subset = df_ml[df_ml["label"] == label_val]["word_count"]
    ax.hist(subset, bins=60, alpha=0.6, color=color, label=name,
            range=(0, min(300, word_cap)))
ax.set_title("Prompt Word Count Distribution", fontsize=14, fontweight="bold", pad=10)
ax.set_xlabel("Number of Words", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/05_word_count_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
word_stats = df_ml.groupby("label")["word_count"].agg(["mean", "median", "max"]).round(1)
print(f"Fig 5 saved | Word count stats:\n{word_stats.to_string()}")

print("\nAll EDA figures saved to figures/")
