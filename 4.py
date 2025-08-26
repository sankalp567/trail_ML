import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Optional TSFEL ----
try:
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:
    TSFEL_AVAILABLE = False
    print("[Info] TSFEL not installed. Install with: pip install tsfel")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_TRAIN = os.path.join(BASE_DIR, "Combined", "Train")
UCI_BASE = os.path.join(BASE_DIR, "UCI HAR Dataset")

# -----------------------------
# Load a subset of samples & compute TSFEL features
# -----------------------------
if TSFEL_AVAILABLE:
    print("[Step] Extracting TSFEL features ...")
    cfg = tsfel.get_features_by_domain()

    # Just pick a few files per activity to keep runtime reasonable
    ACTIVITIES = os.listdir(COMBINED_TRAIN)
    feats_list = []
    for act in ACTIVITIES:
        folder = os.path.join(COMBINED_TRAIN, act)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".txt", ".csv"))])[:3]
        for f in files:
            df = pd.read_csv(os.path.join(folder, f), sep=None, engine="python")
            if df.shape[1] == 1:
                df = df.iloc[:, 0].str.split(",", expand=True)
            df = df.apply(pd.to_numeric, errors="coerce").dropna().iloc[:, :3]
            df.columns = ["accx", "accy", "accz"]

            feats = tsfel.time_series_features_extractor(cfg, df, fs=50)
            feats_list.append(feats)

    tsfel_features = pd.concat(feats_list, axis=0).reset_index(drop=True)

    print(f"[OK] TSFEL feature matrix shape: {tsfel_features.shape}")

    # Correlation matrix
    corr_tsfel = tsfel_features.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_tsfel, cmap="coolwarm", center=0, cbar=False)
    plt.title("Correlation Matrix - TSFEL Features")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Load UCI HAR 561 Features
# -----------------------------
print("[Step] Loading UCI HAR provided features ...")
X_train = pd.read_csv(
    os.path.join(UCI_BASE, "train", "X_train.txt"),
    sep=r"\s+", header=None, engine="python"
)
features = pd.read_csv(
    os.path.join(UCI_BASE, "features.txt"),
    sep=r"\s+", header=None, engine="python", index_col=0
)

X_train.columns = features[1]

print(f"[OK] UCI feature matrix shape: {X_train.shape}")

# Correlation matrix (sample a subset of features for visualization)
subset = X_train.iloc[:, :50]  # only first 50 for clearer heatmap
corr_provided = subset.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_provided, cmap="coolwarm", center=0, cbar=False)
plt.title("Correlation Matrix - Provided 561 Features (subset shown)")
plt.tight_layout()
plt.show()
