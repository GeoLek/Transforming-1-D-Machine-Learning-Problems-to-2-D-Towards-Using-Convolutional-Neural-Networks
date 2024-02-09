import pandas as pd
import os

def segment_ecg_around_qrs(ecg_file_path, qrs_file_path, segment_length, output_folder, fs):
    # Load ECG data
    ecg_data = pd.read_csv(ecg_file_path)
    # Round 'time_ms' in ECG data to 8 decimal places
    ecg_data['time_ms'] = ecg_data['time_ms'].round(8)

    # Load QRS complex locations
    qrs_data = pd.read_csv(qrs_file_path)
    # Round 'time_ms' in QRS data to 8 decimal places
    qrs_data['time_ms'] = qrs_data['time_ms'].round(8)

    # Iterate over each QRS location and extract the segment
    for index, row in qrs_data.iterrows():
        qrs_time = row['time_ms']
        # Find the nearest time in the ECG data
        nearest_time_index = (ecg_data['time_ms'] - qrs_time).abs().idxmin()
        start = max(0, nearest_time_index - segment_length // 2)
        end = min(len(ecg_data), nearest_time_index + segment_length // 2)

        segment = ecg_data.iloc[start:end]

        # Save the segment to a new CSV file or process it further
        segment_file_name = f'{os.path.splitext(os.path.basename(ecg_file_path))[0]}_segment_{index}.csv'
        segment.to_csv(os.path.join(output_folder, segment_file_name), index=False)

# Parameters
segment_length = 100  # in samples
fs = 360  # Sampling frequency

# Paths
ecg_dataset_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/Processed CSV files'
qrs_dataset_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Pan–Tompkins algorithm/QRS complexes'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results'

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each ECG record
for record_number in range(100, 235):
    ecg_file_path = os.path.join(ecg_dataset_path, f'{record_number}.csv')
    qrs_file_path = os.path.join(qrs_dataset_path, f'{record_number}_MLII_QRS.csv')

    if os.path.exists(ecg_file_path) and os.path.exists(qrs_file_path):
        segment_ecg_around_qrs(ecg_file_path, qrs_file_path, segment_length, output_folder, fs)
