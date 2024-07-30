import pandas as pd
import numpy as np
import os

def segment_ecg_around_qrs(ecg_file_path, qrs_file_path, annotation_file_path, segment_length, output_folder):
    # Load ECG data
    ecg_data = pd.read_csv(ecg_file_path)
    # Standardize column names to remove any extra spaces
    ecg_data.columns = ecg_data.columns.str.strip()
    # Round 'Time (ms)' in ECG data to 8 decimal places
    ecg_data['Time (ms)'] = ecg_data['Time (ms)'].round(8)

    # Load QRS complex locations
    qrs_data = pd.read_csv(qrs_file_path)
    # Standardize column names to remove any extra spaces
    qrs_data.columns = qrs_data.columns.str.strip()
    # Round 'Time (ms)' in QRS data to 8 decimal places
    qrs_data['Time (ms)'] = qrs_data['Time (ms)'].round(8)

    # Load annotation data
    annotation_data = pd.read_csv(annotation_file_path)
    # Standardize column names to remove any extra spaces
    annotation_data.columns = annotation_data.columns.str.strip()
    # Round 'Time (ms)' in annotation data to 8 decimal places
    annotation_data['Time (ms)'] = annotation_data['Time (ms)'].round(8)

    # Print column names for debugging
    print(f"ECG data columns: {ecg_data.columns}")
    print(f"QRS data columns: {qrs_data.columns}")
    print(f"Annotation data columns: {annotation_data.columns}")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Create an empty DataFrame to store the segments
    segments_df = pd.DataFrame(columns=['Sample index', 'Time (ms)', 'Symbol', 'Description', 'Channels', 'MLII', 'V1'])

    # Iterate over each QRS location and extract the segment
    for index, row in qrs_data.iterrows():
        qrs_time = row['Time (ms)']
        # Find the nearest time in the ECG data
        nearest_time_index = (ecg_data['Time (ms)'] - qrs_time).abs().idxmin()
        start = max(0, nearest_time_index - segment_length // 2)
        end = min(len(ecg_data), nearest_time_index + segment_length // 2)

        segment = ecg_data.iloc[start:end].copy()

        # Find the matching annotation for the current QRS time
        annotation_row = annotation_data.loc[(annotation_data['Time (ms)'] - qrs_time).abs().idxmin()]

        segment['Symbol'] = annotation_row.get('Symbol', 'Unknown')
        segment['Description'] = annotation_row.get('Description', 'Unknown')
        segment['Channels'] = annotation_row.get('Channels', 'Unknown')
        segment['Sample index'] = row.get('Sample index', index)  # Use index if Sample index is not available

        segments_df = pd.concat([segments_df, segment], ignore_index=True)

    # Save the segments to a new CSV file
    segment_file_name = f'{os.path.splitext(os.path.basename(ecg_file_path))[0]}_annotation_details_segments.csv'
    segments_df.to_csv(os.path.join(output_folder, segment_file_name), index=False)


# Parameters
segment_length = 100  # in samples
fs = 360  # Sampling frequency

# Path to the dataset folder
dataset_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Extracted_Processed_Beats'
qrs_output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Pan-Tompkins_algorithm_output'
annotation_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/MIT-BIH Arrhythmia Database/mit-bih-arrhythmia-database-1.0.0'
segment_output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Segmented_Beats'

# Ensure the output directory exists
os.makedirs(segment_output_path, exist_ok=True)

# Get a list of all CSV files in the dataset folder
csv_files = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]

# Process each file and extract segments around QRS locations
for csv_file in csv_files:
    ecg_file_path = os.path.join(dataset_path, csv_file)
    record_name = os.path.splitext(csv_file)[0]

    # Search for any QRS file that starts with the record name
    qrs_files = [file for file in os.listdir(qrs_output_path) if
                 file.startswith(record_name) and file.endswith('_QRS.csv')]
    annotation_file_path = os.path.join(annotation_path, f'{record_name}_annotation_details.csv')

    if qrs_files and os.path.exists(annotation_file_path):
        for qrs_file in qrs_files:
            qrs_file_path = os.path.join(qrs_output_path, qrs_file)
            print(f"Found QRS file: {qrs_file_path}")
            segment_ecg_around_qrs(ecg_file_path, qrs_file_path, annotation_file_path, segment_length, segment_output_path)
    else:
        print(f'QRS file or annotation file for {csv_file} not found. Skipping.')

print("Segment extraction complete.")
