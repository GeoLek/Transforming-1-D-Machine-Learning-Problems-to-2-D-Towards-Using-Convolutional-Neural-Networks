import wfdb
import numpy as np
import pandas as pd
import os
from scipy.signal import medfilt, detrend

# Define the directory containing the MIT-BIH data files
data_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/MIT-BIH Arrhythmia Database/mit-bih-arrhythmia-database-1.0.0'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Extracted_Processed_Beats'  # Specify your output directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Desired annotations to check
desired_annotations = ['N', 'S', 'V', 'F', 'Q']

# Preprocessing functions
def filter_noise(ecg_signal, kernel_size=3):
    return medfilt(ecg_signal, kernel_size=kernel_size)

def remove_baseline_wander(ecg_signal):
    return detrend(ecg_signal, type='linear')

def normalize_signal(ecg_signal):
    mean_val = np.mean(ecg_signal)
    std_val = np.std(ecg_signal)
    return (ecg_signal - mean_val) / std_val

def preprocess_ecg_signal(ecg_data):
    # Preprocess each lead in the DataFrame
    for column in ecg_data.columns[5:]:  # Skipping columns to process the signals
        ecg_data[column] = filter_noise(ecg_data[column])
        ecg_data[column] = remove_baseline_wander(ecg_data[column])
        ecg_data[column] = normalize_signal(ecg_data[column])
    return ecg_data

# Function to extract beats for a single record
def extract_beats(record, annotations, signal, time_ms, window_size=100):
    beat_segments = []
    ann_sample_indices = annotations.sample
    ann_symbols = annotations.symbol

    for idx, symbol in zip(ann_sample_indices, ann_symbols):
        if idx - window_size >= 0 and idx + window_size < len(signal):
            beat_segment = signal[idx - window_size: idx + window_size + 1]
            beat_time_ms = time_ms[idx - window_size: idx + window_size + 1]
            beat_segments.append((beat_segment, beat_time_ms, symbol))

    return beat_segments

# Function to process a single record and save additional details
def process_record(record_number):
    record_name = f'{record_number:03d}'  # Format record number with leading zeros

    try:
        # Load the header file to get channel names
        header = wfdb.rdheader(os.path.join(data_dir, record_name))
        print(f"Columns from header file for record {record_name}: {header.sig_name}")

        # Load the ECG signal data
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        signal = record.p_signal[:, 0]  # Assuming we want the first channel (e.g., MLII)

        # Calculate the "time_ms" column based on the sampling frequency
        sampling_frequency = header.fs  # Sampling frequency in Hz
        num_samples = len(record.p_signal)
        time_ms = [1000 * i / sampling_frequency for i in range(num_samples)]

        # Create a DataFrame with the signal data, "Sample index," and "Time (ms)"
        ecg_data = pd.DataFrame({'Sample index': range(num_samples), 'Time (ms)': time_ms})
        for i in range(header.n_sig):
            ecg_data[header.sig_name[i]] = record.p_signal[:, i]

        print(f"Columns in the ecg_data DataFrame for record {record_name}: {ecg_data.columns}")

        # Preprocess the ECG signals
        ecg_data = preprocess_ecg_signal(ecg_data)

        # Read the annotations
        annotations = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
        print(f"Annotation columns for record {record_name}: {annotations.__dict__}")

        # Extract annotation sample indices and types
        ann_sample_indices = annotations.sample  # The indices of the annotations in the signal
        ann_symbols = annotations.symbol  # The annotation symbols (e.g., types of beats)

        # Map annotation symbols to human-readable descriptions
        annotation_map = {
            'N': 'Normal beat',
            'S': 'Supraventricular premature beat',
            'V': 'Premature ventricular contraction',
            'F': 'Fusion of ventricular and normal beat',
            'Q': 'Unclassifiable beat'
        }

        # Initialize annotation and description columns with empty strings
        ecg_data['Symbol'] = ''
        ecg_data['Description'] = ''

        # Add annotations and descriptions to the DataFrame
        for idx, symbol in zip(ann_sample_indices, ann_symbols):
            ecg_data.at[idx, 'Symbol'] = symbol
            ecg_data.at[idx, 'Description'] = annotation_map.get(symbol, 'Unknown')

        # Add the 'Channels' column
        ecg_data['Channels'] = ', '.join(header.sig_name)

        # Collect annotation details after preprocessing
        annotation_details = []
        for idx, symbol in zip(ann_sample_indices, ann_symbols):
            annotation_info = {
                'Sample index': idx,
                'Time (ms)': time_ms[idx],
                'Symbol': symbol,
                'Description': annotation_map.get(symbol, 'Unknown'),
                'Channels': ', '.join(header.sig_name)
            }
            for sig_name in header.sig_name:
                annotation_info[sig_name] = ecg_data.at[idx, sig_name]
            annotation_details.append(annotation_info)

        # Convert annotation details to DataFrame
        annotation_details_df = pd.DataFrame(annotation_details)

        # Save the main DataFrame as a CSV file in the output directory
        csv_filename = os.path.join(output_dir, f'{record_name}.csv')
        ecg_data.to_csv(csv_filename, index=False)

        print(f'Record {record_name} saved as {csv_filename}')

        # Save the annotation details to an additional CSV file
        annotation_csv_filename = os.path.join(output_dir, f'{record_name}_annotation_details.csv')
        print(f"Saving annotation details to {annotation_csv_filename} with the first 5 rows:\n{annotation_details_df.head()}")
        annotation_details_df.to_csv(annotation_csv_filename, index=False)

        # Extract beats and save to a separate CSV file
        beat_segments = extract_beats(record, annotations, signal, time_ms)

        # Create a DataFrame where each column represents a sample in the beat segment
        max_len = max(len(seg[0]) for seg in beat_segments)
        beat_data = {
            'Symbol': [seg[2] for seg in beat_segments]
        }
        for i in range(max_len):
            beat_data[f'Sample_{i}'] = [seg[0][i] if i < len(seg[0]) else np.nan for seg in beat_segments]
            beat_data[f'Time_ms_{i}'] = [seg[1][i] if i < len(seg[1]) else np.nan for seg in beat_segments]

        beats_df = pd.DataFrame(beat_data)
        beats_csv_filename = os.path.join(output_dir, f'{record_name}_beats.csv')
        beats_df.to_csv(beats_csv_filename, index=False)

        print(f'Extracted beats for record {record_name} and saved to {beats_csv_filename}')

    except FileNotFoundError:
        print(f'Record {record_name} not found. Skipping...')

# Loop through each record number and process the files
for record_number in record_numbers:
    process_record(record_number)

print("Processing complete.")
