import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, convolve
import os

# Pan-Tompkins algorithm functions
def bandpass_filter(data, lowcut=5.0, highcut=15.0, signal_freq=360, filter_order=1):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def derivative(signal):
    # Approximate the derivative of the signal with a difference operation
    return np.diff(signal)

def squaring(signal):
    return signal ** 2

def moving_window_integration(signal, window_size=15):
    window = np.ones(window_size) / window_size
    return convolve(signal, window, mode='same')

def findpeaks(data, spacing=1, limit=None):
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]
        start = spacing
        h_c = x[start: start + len]
        start = spacing + s + 1
        h_a = x[start: start + len]
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind

def pan_tompkins(signal, fs=360):
    # Ensure the signal is a NumPy array of type float
    signal = np.asarray(signal, dtype=float)
    filtered_ecg = bandpass_filter(signal, lowcut=5.0, highcut=15.0, signal_freq=fs, filter_order=1)
    diff_ecg = derivative(filtered_ecg)
    squared_ecg = squaring(diff_ecg)
    integrated_ecg = moving_window_integration(squared_ecg, window_size=int(fs / 24))
    peak_indices = findpeaks(integrated_ecg, spacing=int(fs / 2.5), limit=None)

    return peak_indices, integrated_ecg

# Load ECG data from a CSV file and run the Pan-Tompkins algorithm
def run_pan_tompkins_on_csv(csv_file_path, output_folder, fs=360):
    df = pd.read_csv(csv_file_path)
    lead_names = df.columns[5:]  # Leads are after 'Sample index', 'Time (ms)', 'Symbol', 'Description', and 'Channels'

    # Standardize column names to remove any extra spaces
    df.columns = df.columns.str.strip()

    # Check the exact name of the time column
    time_column = [col for col in df.columns if 'time' in col.lower()][0]

    for lead_name in lead_names:
        # Convert the signal to numeric values, ignoring non-numeric values
        ecg_signal = pd.to_numeric(df[lead_name], errors='coerce').dropna().values

        if len(ecg_signal) == 0:
            print(f"No valid numeric data in lead {lead_name} of file {csv_file_path}. Skipping.")
            continue

        peak_indices, processed_ecg = pan_tompkins(ecg_signal, fs)

        # Extract QRS complex amplitude and time
        qrs_amplitudes = processed_ecg[peak_indices]
        qrs_times = df.loc[peak_indices, time_column].values

        # Save the QRS amplitudes and corresponding times to a CSV file
        qrs_df = pd.DataFrame({'QRS_Amplitude': qrs_amplitudes, 'Time (ms)': qrs_times})
        qrs_filename = os.path.splitext(os.path.basename(csv_file_path))[0] + f'_{lead_name}_QRS.csv'
        qrs_df.to_csv(os.path.join(output_folder, qrs_filename), index=False)

        # Plotting the results
        plt.figure(figsize=(12, 8))
        # Plot original ECG Signal with time on x-axis
        plt.subplot(211)
        plt.plot(df[time_column], df[lead_name])
        plt.title(f'Original ECG Signal - {lead_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')

        # Plot Processed ECG Signal with Detected Peaks
        plt.subplot(212)
        plt.plot(df[time_column][:-1], processed_ecg)  # processed_ecg is shorter by one due to np.diff in derivative
        plt.scatter(df[time_column][peak_indices], processed_ecg[peak_indices], color='red', label='QRS Peaks')
        plt.title(f'Processed ECG Signal with Detected Peaks - {lead_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Integrated Amplitude')
        plt.legend()

        # Save the plot
        plot_filename = os.path.splitext(os.path.basename(csv_file_path))[0] + f'_{lead_name}.png'
        plt.savefig(os.path.join(output_folder, plot_filename))
        plt.close()

# Path to the dataset folder
dataset_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Extracted_Processed_Beats'
output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Pan-Tompkins_algorithm_output'


# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Get a list of all CSV files in the dataset folder
csv_files = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]

# Process each file and save QRS locations and times
for csv_file in csv_files:
    csv_file_path = os.path.join(dataset_path, csv_file)
    run_pan_tompkins_on_csv(csv_file_path, output_path)

print("Processing complete.")