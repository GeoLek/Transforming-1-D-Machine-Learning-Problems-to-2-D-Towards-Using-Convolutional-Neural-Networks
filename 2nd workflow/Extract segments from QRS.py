import pandas as pd
import numpy as np
import os


def segment_ecg_around_qrs(ecg_file_path, qrs_file_path, segment_length, output_folder):
    # Load ECG data
    ecg_data = pd.read_csv(ecg_file_path)
    # Standardize column names to remove any extra spaces
    ecg_data.columns = ecg_data.columns.str.strip()
    # Round 'time_ms' in ECG data to 8 decimal places
    ecg_data['Time (ms)'] = ecg_data['Time (ms)'].round(8)

    # Load QRS complex locations
    qrs_data = pd.read_csv(qrs_file_path)
    # Round 'Time (ms)' in QRS data to 8 decimal places
    qrs_data['Time (ms)'] = qrs_data['Time (ms)'].round(8)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each QRS location and extract the segment
    for index, row in qrs_data.iterrows():
        qrs_time = row['Time (ms)']
        # Find the nearest time in the ECG data
        nearest_time_index = (ecg_data['Time (ms)'] - qrs_time).abs().idxmin()
        start = max(0, nearest_time_index - segment_length // 2)
        end = min(len(ecg_data), nearest_time_index + segment_length // 2)

        segment = ecg_data.iloc[start:end]

        # Save the segment to a new CSV file or process it further
        segment_file_name = f'{os.path.splitext(os.path.basename(ecg_file_path))[0]}_segment_{index}.csv'
        segment.to_csv(os.path.join(output_folder, segment_file_name), index=False)


# Parameters
segment_length = 100  # in samples
fs = 360  # Sampling frequency

# Path to the dataset folder
dataset_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Extracted_Processed_Beats'
qrs_output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Pan-Tompkins_algorithm_output'
segment_output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Segmented_Beats'

# Get a list of all CSV files in the dataset folder
csv_files = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]

# Process each file and extract segments around QRS locations
for csv_file in csv_files:
    ecg_file_path = os.path.join(dataset_path, csv_file)
    qrs_file_name = f"{os.path.splitext(csv_file)[0]}_QRS.csv"
    qrs_file_path = os.path.join(qrs_output_path, qrs_file_name)

    if os.path.exists(qrs_file_path):
        segment_ecg_around_qrs(ecg_file_path, qrs_file_path, segment_length, segment_output_path)
    else:
        print(f'QRS file for {csv_file} not found. Skipping.')

print("Segment extraction complete.")
