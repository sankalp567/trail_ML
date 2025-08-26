import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# ---- Optional TSFEL ----
try:
    import tsfel
    TSFEL_AVAILABLE = True
except Exception:
    TSFEL_AVAILABLE = False
    print("[Info] TSFEL not installed. Install with: pip install tsfel")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_TRAIN = os.path.join(BASE_DIR, "Combined", "Train")
COMBINED_TEST = os.path.join(BASE_DIR, "Combined", "Test")
UCI_BASE = os.path.join(BASE_DIR, "UCI HAR Dataset")

ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]
TARGET_LEN = 256

# -----------------------------
# Utility functions
# -----------------------------
def resample_to_len(arr_1d, target_len):
    n = len(arr_1d)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr_1d)

def resample_multichannel(arr_2d, target_len):
    T, C = arr_2d.shape
    out = np.zeros((target_len, C))
    for c in range(C):
        out[:, c] = resample_to_len(arr_2d[:, c], target_len)
    return out

def read_signal_file(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")

    if df.shape[1] == 1:
        col0 = df.iloc[:, 0].astype(str)
        if col0.str.contains(",").any():
            df = col0.str.split(",", expand=True)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 3:
        for _ in range(3 - df.shape[1]):
            df[df.shape[1]] = 0.0
    return df.iloc[:, :3].values

def load_combined_flattened(base_path, target_len=256):
    X, y = [], []
    for act in ACTIVITIES:
        folder = os.path.join(base_path, act)
        for f in os.listdir(folder):
            if f.lower().endswith((".txt", ".csv")):
                arr = read_signal_file(os.path.join(folder, f))
                if arr.shape[0] < 5:
                    continue
                arr_rs = resample_multichannel(arr, target_len)
                X.append(arr_rs.flatten())
                y.append(act)
    return np.vstack(X), np.array(y)

def evaluate_model(model, X_train, y_train, X_test, y_test, title, labels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=labels))

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.show()
    return acc, prec, rec

# -----------------------------
# 1. Raw Accelerometer Data
# -----------------------------
print("[Step 1] Raw Accelerometer Data …")
X_train_raw, y_train_raw = load_combined_flattened(COMBINED_TRAIN, TARGET_LEN)
X_test_raw, y_test_raw = load_combined_flattened(COMBINED_TEST, TARGET_LEN)

clf_raw = DecisionTreeClassifier(random_state=42)
acc_raw, prec_raw, rec_raw = evaluate_model(
    clf_raw, X_train_raw, y_train_raw, X_test_raw, y_test_raw,
    "Decision Tree (Raw Accelerometer)", ACTIVITIES
)

# -----------------------------
# 2. TSFEL Features
# -----------------------------
if TSFEL_AVAILABLE:
    print("[Step 2] TSFEL Features …")
    cfg = tsfel.get_features_by_domain()

    def extract_tsfel_features(X_flat, y):
        feats_list, labels = [], []
        T3 = X_flat.shape[1] // 3
        for i in range(len(y)):
            arr = X_flat[i].reshape(T3, 3)
            df = pd.DataFrame(arr, columns=["accx", "accy", "accz"])
            feats = tsfel.time_series_features_extractor(cfg, df, fs=50)
            feats_list.append(feats.values.flatten())
            labels.append(y[i])
        return np.vstack(feats_list), np.array(labels)

    X_train_tsfel, y_train_tsfel = extract_tsfel_features(X_train_raw, y_train_raw)
    X_test_tsfel, y_test_tsfel = extract_tsfel_features(X_test_raw, y_test_raw)

    clf_tsfel = DecisionTreeClassifier(random_state=42)
    acc_tsfel, prec_tsfel, rec_tsfel = evaluate_model(
        clf_tsfel, X_train_tsfel, y_train_tsfel, X_test_tsfel, y_test_tsfel,
        "Decision Tree (TSFEL Features)", ACTIVITIES
    )
else:
    acc_tsfel = prec_tsfel = rec_tsfel = 0
    print("[Skip] TSFEL not installed; skipping TSFEL model.")

# -----------------------------
# 3. Provided 561 Features
# -----------------------------
print("[Step 3] Provided 561 Features …")
X_train = pd.read_csv(os.path.join(UCI_BASE, "train", "X_train.txt"),
                      sep=r"\s+", header=None, engine="python")
y_train = pd.read_csv(os.path.join(UCI_BASE, "train", "y_train.txt"),
                      sep=r"\s+", header=None, engine="python")

X_test = pd.read_csv(os.path.join(UCI_BASE, "test", "X_test.txt"),
                     sep=r"\s+", header=None, engine="python")
y_test = pd.read_csv(os.path.join(UCI_BASE, "test", "y_test.txt"),
                     sep=r"\s+", header=None, engine="python")

activity_labels = pd.read_csv(os.path.join(UCI_BASE, "activity_labels.txt"),
                              sep=r"\s+", header=None, index_col=0, engine="python")
y_train = y_train[0].map(activity_labels[1])
y_test = y_test[0].map(activity_labels[1])

clf_provided = DecisionTreeClassifier(random_state=42)
acc_provided, prec_provided, rec_provided = evaluate_model(
    clf_provided, X_train, y_train, X_test, y_test,
    "Decision Tree (561 Provided Features)", activity_labels[1].values
)

# -----------------------------
# Comparison Summary
# -----------------------------
print("\n=== Summary of Results ===")
print(f"Raw Accelerometer   -> Acc: {acc_raw:.4f}, Prec: {prec_raw:.4f}, Rec: {rec_raw:.4f}")
print(f"TSFEL Features      -> Acc: {acc_tsfel:.4f}, Prec: {prec_tsfel:.4f}, Rec: {rec_tsfel:.4f}")
print(f"561 Provided Feats  -> Acc: {acc_provided:.4f}, Prec: {prec_provided:.4f}, Rec: {rec_provided:.4f}")
