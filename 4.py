# har_task1_problem3_all.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- Optional niceties (progress bars) ----
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# ---- Optional TSFEL (pip install tsfel) ----
try:
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:
    TSFEL_AVAILABLE = False
    print("[Info] TSFEL not installed; install with: pip install tsfel")

# -----------------------------
# Paths & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_TRAIN = os.path.join(BASE_DIR, "Combined", "Train")
UCI_BASE = os.path.join(BASE_DIR, "UCI HAR Dataset")

ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]

TARGET_LEN = 256           # resample each sequence to same length for PCA
VALID_EXTS = (".txt", ".csv")   # add more if needed, e.g. ".dat"

# -----------------------------
# Utilities
# -----------------------------
def resample_to_len(arr_1d, target_len):
    n = len(arr_1d)
    if n == target_len:
        return arr_1d
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, arr_1d)

def resample_multichannel(arr_2d, target_len):
    T, C = arr_2d.shape
    out = np.zeros((target_len, C), dtype=float)
    for c in range(C):
        out[:, c] = resample_to_len(arr_2d[:, c], target_len)
    return out

def read_signal_file(path):
    """
    Robust reader for Combined CSV/TXT files.
    Handles header 'accx,accy,accz', comma/space delimiter, and stray text rows.
    Returns numeric (T, 3) ndarray in order [accx, accy, accz].
    """
    # Try auto-delimiter
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")

    # If single column that actually contains commas, split it
    if df.shape[1] == 1:
        col0 = df.iloc[:, 0].astype(str)
        if col0.str.contains(",").any():
            df = col0.str.split(",", expand=True)

    # Coerce to numeric and drop non-numeric rows
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    # Ensure at least 3 columns (pad if needed), then take first 3
    if df.shape[1] < 3:
        for _ in range(3 - df.shape[1]):
            df[df.shape[1]] = 0.0
    df = df.iloc[:, :3]

    arr = df.values
    return arr

def load_combined_flattened_all(base_path, target_len=256):
    """
    Loads ALL files for each activity from Combined/Train.
    Resamples to target_len and flattens to 1D.
    Returns X (N, target_len*3), y (labels).
    """
    X, y = [], []
    for act in ACTIVITIES:
        folder = os.path.join(base_path, act)
        if not os.path.isdir(folder):
            print(f"[Warn] Missing activity folder: {folder}")
            continue

        files = [f for f in sorted(os.listdir(folder))
                 if os.path.isfile(os.path.join(folder, f))
                 and f.lower().endswith(VALID_EXTS)]

        iterator = tqdm(files, desc=f"Loading {act}") if TQDM else files
        for f in iterator:
            fp = os.path.join(folder, f)
            arr = read_signal_file(fp)        # (T, >=3)
            if arr.shape[0] < 5:
                continue
            arr = arr[:, :3]                  # keep 3 axes
            arr_rs = resample_multichannel(arr, target_len)   # (target_len, 3)
            X.append(arr_rs.flatten())        # (target_len*3,)
            y.append(act)

    if len(X) == 0:
        raise RuntimeError("No samples loaded from Combined dataset.")
    return np.vstack(X), np.array(y)

def plot_pca(X, y, title):
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(7.5, 6.2))
    for lab in np.unique(y):
        idx = (y == lab)
        plt.scatter(Xp[idx, 0], Xp[idx, 1], s=16, alpha=0.7, label=lab)
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()

# -----------------------------
# PCA #1 — Raw total acceleration (ALL samples in Combined/Train)
# -----------------------------
print("[Step] Loading ALL raw signals from Combined/Train …")
X_raw, y_raw = load_combined_flattened_all(COMBINED_TRAIN, target_len=TARGET_LEN)
print(f"[OK] Raw matrix shape: {X_raw.shape} | samples per class:",
      {lab: int((y_raw == lab).sum()) for lab in np.unique(y_raw)})
plot_pca(X_raw, y_raw, "PCA — Raw Total Acceleration (accx, accy, accz)")

# -----------------------------
# PCA #2 — TSFEL features for ALL samples (if tsfel available)
# -----------------------------
if TSFEL_AVAILABLE:
    print("[Step] Extracting TSFEL features for ALL raw samples …")
    cfg = tsfel.get_features_by_domain()
    X_tsfel_list, y_tsfel_list = [], []

    iterator = tqdm(range(len(y_raw)), desc="TSFEL") if TQDM else range(len(y_raw))
    # Rebuild each sample to (T,3) then extract features
    T3 = X_raw.shape[1] // 3
    for i in iterator:
        arr = X_raw[i].reshape(T3, 3)
        df = pd.DataFrame(arr, columns=["accx", "accy", "accz"])
        feats = tsfel.time_series_features_extractor(cfg, df, fs=50)
        X_tsfel_list.append(feats.values.flatten())
        y_tsfel_list.append(y_raw[i])

    X_tsfel = np.vstack(X_tsfel_list)
    y_tsfel = np.array(y_tsfel_list)
    print(f"[OK] TSFEL matrix shape: {X_tsfel.shape} | samples per class:",
          {lab: int((y_tsfel == lab).sum()) for lab in np.unique(y_tsfel)})
    plot_pca(X_tsfel, y_tsfel, "PCA — TSFEL Features (ALL samples)")
else:
    print("[Skip] TSFEL not installed; skipping PCA on TSFEL features.")

# -----------------------------
# PCA #3 — 561 engineered features (ALL train samples in UCI HAR Dataset)
# -----------------------------
print("[Step] Loading ALL UCI HAR engineered features (train) …")
X_train = pd.read_csv(
    os.path.join(UCI_BASE, "train", "X_train.txt"),
    sep=r"\s+", header=None, engine="python"
)
y_train = pd.read_csv(
    os.path.join(UCI_BASE, "train", "y_train.txt"),
    sep=r"\s+", header=None, engine="python"
)
activity_labels = pd.read_csv(
    os.path.join(UCI_BASE, "activity_labels.txt"),
    sep=r"\s+", header=None, index_col=0, engine="python"
)
y_train = y_train[0].map(activity_labels[1])

print(f"[OK] UCI 561-feature matrix: {X_train.shape} | samples per class:",
      {lab: int((y_train.values == lab).sum()) for lab in y_train.unique()})
plot_pca(X_train.values, y_train.values, "PCA — Provided 561 Features (ALL train samples)")

