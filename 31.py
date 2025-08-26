import os
import math
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_BASE = os.path.join(BASE_DIR, "MyRawPhoneData")      # input root
OUT_BASE = os.path.join(BASE_DIR, "Processed", "Combined", "Train")  # output root

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]

TARGET_HZ = 50
TARGET_SEC = 10
TARGET_N = TARGET_HZ * TARGET_SEC  # 500 rows

VALID_EXTS = (".csv", ".txt")

def discover_columns(df):
    """Try to find time and accel columns with flexible headers."""
    cols = [c.lower() for c in df.columns]

    # time
    time_candidates = [i for i,c in enumerate(cols) if "time" in c or "timestamp" in c]
    tcol = time_candidates[0] if time_candidates else None

    # accel columns likely named with x/y/z
    # fallbacks for common names
    ax_idx = next((i for i,c in enumerate(cols) if "x" == c or "accx" in c or "ax" == c or " accel x" in c or ("x" in c and "acc" in c)), None)
    ay_idx = next((i for i,c in enumerate(cols) if "y" == c or "accy" in c or "ay" == c or " accel y" in c or ("y" in c and "acc" in c)), None)
    az_idx = next((i for i,c in enumerate(cols) if "z" == c or "accz" in c or "az" == c or " accel z" in c or ("z" in c and "acc" in c)), None)

    # If still missing, try to pick first three numeric columns (excluding time)
    if ax_idx is None or ay_idx is None or az_idx is None:
        num_df = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = [i for i in range(len(df.columns)) if num_df.iloc[:, i].notna().mean() > 0.95]
        if tcol is not None and tcol in numeric_cols:
            numeric_cols.remove(tcol)
        if len(numeric_cols) >= 3:
            ax_idx, ay_idx, az_idx = numeric_cols[:3]

    return tcol, ax_idx, ay_idx, az_idx

def load_raw_file(path):
    # try auto delimiter first
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, header=0, sep=r"\s+", engine="python")
    return df

def to_mps2_if_needed(ax, ay, az):
    """Heuristic: if typical magnitude ~1–3, assume 'g' and convert to m/s^2."""
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    median_mag = np.nanmedian(mag)
    if 0.5 <= median_mag <= 3.5:  # likely in g
        g = 9.81
        return ax * g, ay * g, az * g, "g->m/s^2"
    return ax, ay, az, "m/s^2"

def central_window_by_time(df, tcol, sec=10):
    """Take central sec seconds using timestamp; if recording shorter, use all."""
    if tcol is None:
        return df  # will handle by index later

    t = df.iloc[:, tcol].astype(float).values
    t = t - t[0]  # start at 0
    df = df.copy()
    df.iloc[:, tcol] = t
    if t[-1] < sec:
        return df

    center = t[-1] / 2.0
    start = center - sec/2.0
    end   = center + sec/2.0
    mask = (t >= start) & (t <= end)
    return df.loc[mask]

def resample_to_fixed_hz(time, ax, ay, az, hz=50, sec=10):
    """Resample to exact hz over sec seconds using linear interpolation."""
    # build target timeline [0, sec]
    t_new = np.linspace(0, sec, hz*sec, endpoint=False)  # 0..(sec-1/hz)
    # ensure time starts at 0 and covers sec; if not, stretch by interpolation on the existing span
    t = np.asarray(time, dtype=float)
    t = t - t[0]
    if t[-1] < 1e-6:  # degenerate
        # synthesize uniform time
        t = np.linspace(0, sec, len(ax), endpoint=False)
    # interpolate
    ax_new = np.interp(t_new, np.clip(t, 0, t[-1]), ax)
    ay_new = np.interp(t_new, np.clip(t, 0, t[-1]), ay)
    az_new = np.interp(t_new, np.clip(t, 0, t[-1]), az)
    return ax_new, ay_new, az_new

def process_one_file(in_path, out_path):
    raw = load_raw_file(in_path)
    # find columns
    tcol, ax_i, ay_i, az_i = discover_columns(raw)
    if ax_i is None or ay_i is None or az_i is None:
        raise ValueError(f"Could not locate accel columns in {in_path}")

    # coerce numeric
    df = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    # center window (if timestamp exists)
    df_c = central_window_by_time(df, tcol, sec=TARGET_SEC)

    # build arrays
    if tcol is not None:
        time = df_c.iloc[:, tcol].astype(float).values
    else:
        # synthesize time from index (assume near 50 Hz; resample will handle)
        time = np.arange(len(df_c)) / float(TARGET_HZ)
    ax = df_c.iloc[:, ax_i].astype(float).values
    ay = df_c.iloc[:, ay_i].astype(float).values
    az = df_c.iloc[:, az_i].astype(float).values

    # unit check/convert
    ax, ay, az, unit_note = to_mps2_if_needed(ax, ay, az)

    # resample to exact 50 Hz & exactly 10 s
    ax, ay, az = resample_to_fixed_hz(time, ax, ay, az, hz=TARGET_HZ, sec=TARGET_SEC)

    # final check length
    if len(ax) != TARGET_N:
        # pad/crop if a tiny mismatch
        ax = np.resize(ax, TARGET_N)
        ay = np.resize(ay, TARGET_N)
        az = np.resize(az, TARGET_N)

    # save standardized CSV
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"accx": ax, "accy": ay, "accz": az}).to_csv(out_path, index=False)

    return unit_note

def main():
    total_in, total_ok = 0, 0
    for act in ACTIVITIES:
        in_dir = os.path.join(RAW_BASE, act)
        if not os.path.isdir(in_dir):
            print(f"[skip] {in_dir} (not found)")
            continue
        out_dir = os.path.join(OUT_BASE, act)
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(in_dir)):
            if not fname.lower().endswith(VALID_EXTS):
                continue
            total_in += 1
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            try:
                unit_note = process_one_file(in_path, out_path)
                total_ok += 1
                print(f"[OK] {act}/{fname} -> {unit_note}")
            except Exception as e:
                print(f"[ERR] {act}/{fname}: {e}")

    print(f"\nDone. {total_ok}/{total_in} files processed successfully.")
    print(f"Output in: {OUT_BASE}")
    print("Each file has 500 rows (10 s @ 50 Hz) with columns: accx, accy, accz.")

if __name__ == "__main__":
    main()
