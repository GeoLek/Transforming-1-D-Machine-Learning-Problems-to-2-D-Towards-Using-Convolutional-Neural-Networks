import pandas as pd
import matplotlib.pyplot as plt

# File Path for a single record
record = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Processed signals/Processed CSV files/100_merged_data.csv"

# Reading ECG Data into a DataFrame
ecg = pd.read_csv(record)

# Preview the ECG Data
print(ecg)

# Calculate the number of samples required for 4 seconds
sampling_frequency = 360  # Replace with the actual sampling frequency
duration_seconds = 4
num_samples_4_seconds = int(sampling_frequency * duration_seconds)

# Plot the Entire 30-Minute EKG for MLII
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"], ecg["MLII"], label="MLII", color="blue")
plt.title("30-minute EKG of Patient 100 (MLII)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot the Entire 30-Minute EKG for V5
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"], ecg["V5"], label="V5", color="green")
plt.title("30-minute EKG of Patient 100 (V5)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot a 4-Second Segment for MLII
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"][:num_samples_4_seconds], ecg["MLII"][:num_samples_4_seconds], label="MLII", color="red")
plt.title("4-Second Segment of Patient 100's EKG (MLII)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot a 4-Second Segment for V5
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"][:num_samples_4_seconds], ecg["V5"][:num_samples_4_seconds], label="V5", color="purple")
plt.title("4-Second Segment of Patient 100's EKG (V5)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot a Mixed Plot for the Entire 30-Minute EKG (MLII and V5)
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"], ecg["MLII"], label="MLII", color="blue")
plt.plot(ecg["time_ms"], ecg["V5"], label="V5", color="green")
plt.title("30-minute EKG of Patient 100 (Mixed Plot)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot a Mixed Plot for the 4-Second Segment (MLII and V5)
plt.figure(figsize=(12, 4))
plt.plot(ecg["time_ms"][:num_samples_4_seconds], ecg["MLII"][:num_samples_4_seconds], label="MLII", color="blue")
plt.plot(ecg["time_ms"][:num_samples_4_seconds], ecg["V5"][:num_samples_4_seconds], label="V5", color="green")
plt.title("4-Second Segment of Patient 100's EKG (Mixed Plot)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
