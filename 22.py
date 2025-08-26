import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ---- Optional TSFEL (pip install tsfel) ----
try:
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:
    TSFEL_AVAILABLE = False
    print("[Info] TSFEL not installed; TSFEL curve will be skipped. Install with: pip install tsfel")

# -----------------------------
# Paths & constants
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_TRAIN = os.path.join(BASE_DIR, "Combined", "Train")
COMBINED_TEST  = os.path.join(BASE_DIR, "Combined", "Test")
UCI_BASE       = os.path.join(BASE_DIR, "UCI HAR Dataset")

ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]
TARGET_LEN = 256  # resample raw sequences to equal length

# -----------------------------
# Utilities (robust loaders)
# -----------------------------
def resample_to_len(arr_1d, target_len):
    n = len(arr_1d)
    if n == target_len:
        return arr_1d
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr_1d)

def resample_multichannel(arr_2d, target_len):
    T, C = arr_2d.shape
    out = np.zeros((target_len, C), dtype=float)
    for c in range(C):
        out[:, c] = resample_to_len(arr_2d[:, c], target_len)
    return out

def read_signal_file(path):
    # auto-detect delimiter and handle header 'accx,accy,accz'
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    if df.shape[1] == 1:
        col0 = df.iloc[:, 0].astype(str)
        if col0.str.contains(",").any():
            df = col0.str.split(",", expand=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.shape[1] < 3:
        for _ in range(3 - df.shape[1]):
            df[df.shape[1]] = 0.0
    return df.iloc[:, :3].values  # (T,3)

def load_combined_flattened(base_path, target_len=256):
    X, y = [], []
    for act in ACTIVITIES:
        folder = os.path.join(base_path, act)
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith((".txt", ".csv")):
                continue
            arr = read_signal_file(os.path.join(folder, f))
            if arr.shape[0] < 5:
                continue
            arr_rs = resample_multichannel(arr, target_len)
            X.append(arr_rs.flatten())
            y.append(act)
    return np.vstack(X), np.array(y)

def extract_tsfel_matrix(X_flat):
    """X_flat: (N, T*3) from Combined; returns TSFEL features matrix."""
    cfg = tsfel.get_features_by_domain()
    feats_list = []
    T3 = X_flat.shape[1] // 3
    for i in range(X_flat.shape[0]):
        arr = X_flat[i].reshape(T3, 3)
        df = pd.DataFrame(arr, columns=["accx", "accy", "accz"])
        feats = tsfel.time_series_features_extractor(cfg, df, fs=50)
        feats_list.append(feats.values.flatten())
    return np.vstack(feats_list)

# -----------------------------
# 1) Raw accelerometer features
# -----------------------------
print("[Raw] Loading Combined/Train and Combined/Test …")
Xtr_raw, ytr_raw = load_combined_flattened(COMBINED_TRAIN, TARGET_LEN)
Xte_raw, yte_raw = load_combined_flattened(COMBINED_TEST,  TARGET_LEN)

# -----------------------------
# 2) TSFEL features (optional)
# -----------------------------
if TSFEL_AVAILABLE:
    print("[TSFEL] Extracting TSFEL features (train)…")
    Xtr_tsfel = extract_tsfel_matrix(Xtr_raw)
    print("[TSFEL] Extracting TSFEL features (test)…")
    Xte_tsfel = extract_tsfel_matrix(Xte_raw)

# -----------------------------
# 3) Provided 561 features
# -----------------------------
print("[Provided] Loading UCI HAR 561 features …")
Xtr_prov = pd.read_csv(os.path.join(UCI_BASE, "train", "X_train.txt"),
                       sep=r"\s+", header=None, engine="python")
ytr_prov = pd.read_csv(os.path.join(UCI_BASE, "train", "y_train.txt"),
                       sep=r"\s+", header=None, engine="python")
Xte_prov = pd.read_csv(os.path.join(UCI_BASE, "test", "X_test.txt"),
                       sep=r"\s+", header=None, engine="python")
yte_prov = pd.read_csv(os.path.join(UCI_BASE, "test", "y_test.txt"),
                       sep=r"\s+", header=None, engine="python")
activity_labels = pd.read_csv(os.path.join(UCI_BASE, "activity_labels.txt"),
                              sep=r"\s+", header=None, index_col=0, engine="python")
ytr_prov = ytr_prov[0].map(activity_labels[1])
yte_prov = yte_prov[0].map(activity_labels[1])

# -----------------------------
# Sweep depths 2..8
# -----------------------------
depths = list(range(2, 9))
acc_raw, acc_tsfel, acc_prov = [], [], []

def run_depth_sweep(Xtr, ytr, Xte, yte, label):
    accs = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        accs.append(accuracy_score(yte, pred))
    print(f"{label} accuracies:", ["{:.3f}".format(a) for a in accs])
    return accs

print("[Sweep] Raw …")
acc_raw = run_depth_sweep(Xtr_raw, ytr_raw, Xte_raw, yte_raw, "Raw")

if TSFEL_AVAILABLE:
    print("[Sweep] TSFEL …")
    acc_tsfel = run_depth_sweep(Xtr_tsfel, ytr_raw, Xte_tsfel, yte_raw, "TSFEL")

print("[Sweep] Provided 561 …")
acc_prov = run_depth_sweep(Xtr_prov, ytr_prov, Xte_prov, yte_prov, "Provided")

# -----------------------------
# Plot accuracy vs depth
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(depths, acc_raw, marker="o", label="Raw accelerometer")
if TSFEL_AVAILABLE:
    plt.plot(depths, acc_tsfel, marker="o", label="TSFEL features")
plt.plot(depths, acc_prov, marker="o", label="561 provided features")
plt.xlabel("Tree depth")
plt.ylabel("Test Accuracy")
plt.title("Decision Tree: Test Accuracy vs Depth")
plt.xticks(depths)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Tiny comparison table
# -----------------------------
def best(accs):
    if not accs: return (None, None)
    i = int(np.argmax(accs))
    return depths[i], accs[i]

d_raw, b_raw = best(acc_raw)
d_tsf, b_tsf = best(acc_tsfel) if TSFEL_AVAILABLE else (None, None)
d_prov, b_prov = best(acc_prov)

print("\n=== Best depth per method (by test accuracy) ===")
print(f"Raw:      depth={d_raw},  acc={b_raw:.4f}")
if TSFEL_AVAILABLE:
    print(f"TSFEL:    depth={d_tsf},  acc={b_tsf:.4f}")
print(f"Provided: depth={d_prov}, acc={b_prov:.4f}")

# Short recommendation for your report:
print("\nRecommendation: You should see the 'Provided 561 features' curve consistently "
      "on top or close to the top; TSFEL next; Raw lowest. Choose the depth that "
      "maximizes test accuracy (often 5–7) while avoiding overfitting at very high depth.")
