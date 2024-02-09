import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import peakutils
import re

# Define the input folder where your ECG data files are located
input_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/"

# Create the output folder if it doesn't exist
output_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/TEST"
os.makedirs(output_folder, exist_ok=True)

# Function for low-pass filtering
def low_pass_filter(signal, cutoff_frequency, sampling_frequency):
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Function for ECG segmentation using R-peak detection
def segment_ecg(ecg_signal, sampling_frequency):
    peaks, _ = find_peaks(ecg_signal, distance=sampling_frequency * 0.8)  # Adjust distance as needed
    return len(peaks)  # Return the number of peaks

def extract_numeric_part(s):
    # Extracts the numeric part from a string (e.g., 'File: 100.csv' => 100)
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return 0

# Create a list to store the results
results = []

# Process each ECG data file in the input folder
for ecg_file in os.listdir(input_folder):
    if ecg_file.endswith('.csv'):
        try:
            # Load ECG data from the CSV file
            file_path = os.path.join(input_folder, ecg_file)
            ecg_data = pd.read_csv(file_path)

            # Define the sampling frequency (assuming it's 360 Hz)
            sampling_frequency = 360

            # Specify the columns for MLII and the leads starting with 'V'
            v_columns = [col for col in ecg_data.columns if col.startswith('V')]

            if 'MLII' in ecg_data.columns:
                # 'MLII' column exists, so process it and one 'V' column
                ml_ii_column = 'MLII'
                v_column = v_columns[0] if v_columns else None  # Get the first 'V' column if it exists

                if v_column:
                    # Step 1: Filtering (Simple Low-Pass Filter)
                    ecg_data[ml_ii_column] = low_pass_filter(ecg_data[ml_ii_column].values,
                                                              cutoff_frequency=50,
                                                              sampling_frequency=sampling_frequency)

                    ecg_data[v_column] = low_pass_filter(ecg_data[v_column].values,
                                                         cutoff_frequency=50,
                                                         sampling_frequency=sampling_frequency)

                    # Step 2: Count the number of peaks for MLII and V columns
                    num_ml_ii_peaks = segment_ecg(ecg_data[ml_ii_column], sampling_frequency)
                    num_v_peaks = segment_ecg(ecg_data[v_column], sampling_frequency)

                    result = f"File: {ecg_file}, MLII Peaks: {num_ml_ii_peaks}, {v_column} Peaks: {num_v_peaks}"
                    results.append(result)

            elif len(v_columns) == 2:
                # Two 'V' columns exist, so process them
                v_column1 = v_columns[0]
                v_column2 = v_columns[1]

                if v_column1 and v_column2:
                    # Step 1: Filtering (Simple Low-Pass Filter)
                    ecg_data[v_column1] = low_pass_filter(ecg_data[v_column1].values,
                                                          cutoff_frequency=50,
                                                          sampling_frequency=sampling_frequency)
                    ecg_data[v_column2] = low_pass_filter(ecg_data[v_column2].values,
                                                          cutoff_frequency=50,
                                                          sampling_frequency=sampling_frequency)

                    # Step 2: Count the number of peaks for both V columns
                    num_v1_peaks = segment_ecg(ecg_data[v_column1], sampling_frequency)
                    num_v2_peaks = segment_ecg(ecg_data[v_column2], sampling_frequency)

                    result = f"File: {ecg_file}, {v_column1} Peaks: {num_v1_peaks}, {v_column2} Peaks: {num_v2_peaks}"
                    results.append(result)

        except FileNotFoundError:
            print(f"File not found for {ecg_file}. Skipping.")
            continue

# Sort the results numerically based on the file name (e.g., 100.csv, 101.csv)
results.sort(key=lambda x: int(extract_numeric_part(x.split(":")[1].strip())))

# Save the results to a text file
text_file_path = os.path.join(output_folder, "results.txt")
with open(text_file_path, "w") as text_file:
    text_file.write("\n".join(results))

print("Results saved to:", text_file_path)