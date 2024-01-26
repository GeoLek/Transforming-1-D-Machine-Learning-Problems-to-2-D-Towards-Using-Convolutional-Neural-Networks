import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Define the input folder where your ECG data files are located
input_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV"

# Define the output folder to save processed figures
output_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Processed signals"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all CSV files in the input folder (assuming they are named 100.csv, 101.csv, etc.)
ecg_files = [f"{i}.csv" for i in range(100, 235)]

# Function for low-pass filtering
def low_pass_filter(signal, cutoff_frequency, sampling_frequency):
    nyquist_frequency = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Function for ECG segmentation using R-peak detection
def segment_ecg(ecg_signal, sampling_frequency):
    peaks, _ = find_peaks(ecg_signal, distance=sampling_frequency * 0.6)  # Adjust distance as needed
    segments = []
    for peak in peaks:
        start = peak - int(sampling_frequency * 0.5)
        end = peak + int(sampling_frequency * 0.5)
        if start >= 0 and end < len(ecg_signal):
            segments.append(ecg_signal[start:end])
    return segments

# Process each ECG data file in the input folder
for ecg_file in ecg_files:
    try:
        # Load ECG data from the CSV file
        file_path = os.path.join(input_folder, ecg_file)
        ecg_data = pd.read_csv(file_path)

        # Detect available columns ('MLII', 'V1', 'V5', etc.)
        available_columns = ecg_data.columns
        ecg_signal_columns = [col for col in available_columns if col.startswith('V')]

        # Define the sampling frequency (assuming it's 360 Hz)
        sampling_frequency = 360

        for ecg_signal_column in ecg_signal_columns:
            # Step 1: Filtering (Simple Low-Pass Filter)
            ecg_data[ecg_signal_column] = low_pass_filter(ecg_data[ecg_signal_column].values,
                                                          cutoff_frequency=50,
                                                          sampling_frequency=sampling_frequency)

            # Step 2: Resampling (to 360 Hz, if not already)
            desired_sampling_frequency = 360
            if sampling_frequency != desired_sampling_frequency:
                resampled_time = np.arange(0, len(ecg_data)) * (1 / desired_sampling_frequency)
                ecg_data['time_ms'] = resampled_time
                ecg_data = ecg_data.resample(str(1 / desired_sampling_frequency) + 'S').mean().interpolate()

            # Step 3: Segmentation (R-Peak Detection)
            ecg_signal_segments = segment_ecg(ecg_data[ecg_signal_column], desired_sampling_frequency)

            # Step 4: Normalize ECG Amplitudes
            def normalize_signal(signal):
                min_val = min(signal)
                max_val = max(signal)
                return (signal - min_val) / (max_val - min_val)

            # Ensure that both MLII and V lead signals have the same length
            min_length = min(len(ecg_signal_segments[0]), len(ecg_signal_segments[1]))

            # Check for missing values and fill with zeros if needed
            ecg_signal_segments = [signal[:min_length] if len(signal) >= min_length else np.zeros(min_length) for signal in ecg_signal_segments]

            # Create a DataFrame for processed signals
            processed_signals = {
                'time_ms': [i * (1 / desired_sampling_frequency) for i in range(min_length)],
                'MLII': ecg_signal_segments[0],  # Assuming MLII is the first lead
                ecg_signal_column: ecg_signal_segments[1]  # Corresponding V lead
            }

            # Save processed signals as separate CSV files
            ml_ii_output_file = os.path.join(output_folder, f"{ecg_file}_MLII_processed.csv")
            v_output_file = os.path.join(output_folder, f"{ecg_file}_{ecg_signal_column}_processed.csv")

            pd.DataFrame({'time_ms': processed_signals['time_ms'], 'MLII': processed_signals['MLII']}).to_csv(ml_ii_output_file, index=False)
            pd.DataFrame({'time_ms': processed_signals['time_ms'], ecg_signal_column: processed_signals[ecg_signal_column]}).to_csv(v_output_file, index=False)

            # Example: Saving figures (you can customize this part)
            plt.figure(figsize=(10, 4))
            plt.plot(ecg_data['time_ms'], ecg_data[ecg_signal_column], label=ecg_signal_column)
            plt.title(f"Processed {ecg_signal_column} ECG")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude")
            output_figure = os.path.join(output_folder, f"{ecg_file}_{ecg_signal_column}.png")
            plt.savefig(output_figure)
            plt.close()

        # Plot MLII for the first 3 seconds
        if 'MLII' in ecg_data.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(ecg_data['time_ms'][:3 * sampling_frequency], ecg_data['MLII'][:3 * sampling_frequency], label='MLII', color='blue')
            plt.title(f"First 3 Seconds of Processed MLII ECG for {ecg_file}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude")
            plt.legend()
            output_figure = os.path.join(output_folder, f"{ecg_file}_MLII_first3s.png")
            plt.savefig(output_figure)
            plt.close()

        # Plot each corresponding V lead for the first 3 seconds
        for ecg_signal_column in ecg_signal_columns:
            plt.figure(figsize=(10, 4))
            plt.plot(ecg_data['time_ms'][:3 * sampling_frequency], ecg_data[ecg_signal_column][:3 * sampling_frequency], label=ecg_signal_column)
            plt.title(f"First 3 Seconds of Processed {ecg_signal_column} ECG for {ecg_file}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude")
            plt.legend()
            output_figure = os.path.join(output_folder, f"{ecg_file}_{ecg_signal_column}_first3s.png")
            plt.savefig(output_figure)
            plt.close()

    except FileNotFoundError:
        print(f"File not found for {ecg_file}. Skipping.")
        continue
