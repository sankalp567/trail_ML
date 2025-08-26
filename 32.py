import pandas as pd
import numpy as np

# Load your file
df = pd.read_csv("2025-08-2614.55.44.csv")

# Extract time & acceleration
time = df["time"].values
ax = df["ax (m/s^2)"].values
ay = df["ay (m/s^2)"].values
az = df["az (m/s^2)"].values

# Define target
TARGET_HZ = 50
TARGET_SEC = 10
N = TARGET_HZ * TARGET_SEC
t_new = np.linspace(0, TARGET_SEC, N, endpoint=False)

# Resample
ax_new = np.interp(t_new, time, ax)
ay_new = np.interp(t_new, time, ay)
az_new = np.interp(t_new, time, az)

# Save to new CSV
out = pd.DataFrame({"accx": ax_new, "accy": ay_new, "accz": az_new})
out.to_csv("processed_sample.csv", index=False)

print(out.shape)
