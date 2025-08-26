import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to Combined Train dataset
base_path = "Combined/Train"

# Activity folders
activities = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# Plot settings
plt.figure(figsize=(18, 10))

for idx, activity in enumerate(activities):
    # Get the first sample file from the folder
    activity_path = os.path.join(base_path, activity)
    sample_file = os.listdir(activity_path)[0]  # Pick first file
    file_path = os.path.join(activity_path, sample_file)

    # Load the data (assuming txt files with numeric values)
    data = pd.read_csv(file_path)

    # Create subplot
    plt.subplot(2, 3, idx + 1)
    plt.plot(data)
    plt.title(activity)
    plt.xlabel("Time Steps")
    plt.ylabel("Sensor Reading")
    plt.grid()

plt.tight_layout()
plt.show()
