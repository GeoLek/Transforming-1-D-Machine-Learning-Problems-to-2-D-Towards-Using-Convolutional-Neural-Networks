import pandas as pd
import matplotlib.pyplot as plt

# Load the preprocessed ECG data from the CSV file
csv_file = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/Processed CSV files/101_merged_data.csv"
ecg_data = pd.read_csv(csv_file)

# Extract relevant columns
time_ms = ecg_data['time_ms']
ml_ii = ecg_data['MLII']
vx = ecg_data['V1']  # Replace 'x' with the appropriate V lead (e.g., V1, V2, V3, ...)

# Create a line plot for MLII
plt.figure(figsize=(10, 4))
plt.plot(time_ms, ml_ii, label='MLII', color='blue')
plt.title("ECG Waveform - MLII")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Create a line plot for Vx
plt.figure(figsize=(10, 4))
plt.plot(time_ms, vx, label=f'Vx', color='red')
plt.title(f"ECG Waveform - V5")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
