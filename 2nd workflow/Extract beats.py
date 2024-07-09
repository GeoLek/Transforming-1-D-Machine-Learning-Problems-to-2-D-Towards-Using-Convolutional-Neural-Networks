import wfdb
import numpy as np
import pandas as pd
import os

# Define the directory containing the MIT-BIH data files
data_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/MIT-BIH Arrhythmia Database/mit-bih-arrhythmia-database-1.0.0'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'  # Specify your output directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Desired annotations to check
desired_annotations = ['N', 'S', 'V', 'F', 'Q']

# Function to extract beats for a single record
def extract_beats(record, annotations, signal, window_size=100):
    beat_segments = []
    ann_sample_indices = annotations.sample
    ann_symbols = annotations.symbol

    for idx, symbol in zip(ann_sample_indices, ann_symbols):
        if idx - window_size >= 0 and idx + window_size < len(signal):
            beat_segment = signal[idx - window_size: idx + window_size + 1]
            beat_segments.append((beat_segment, symbol))

    return beat_segments

# Function to process a single record and save additional details
def process_record(record_number):
    record_name = f'{record_number:03d}'  # Format record number with leading zeros

    try:
        # Load the header file to get channel names
        header = wfdb.rdheader(os.path.join(data_dir, record_name))

        # Load the ECG signal data
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))
        signal = record.p_signal[:, 0]  # Assuming we want the first channel (e.g., MLII)

        # Calculate the "time_ms" column based on the sampling frequency
        sampling_frequency = header.fs  # Sampling frequency in Hz
        num_samples = len(record.p_signal)
        time_ms = [1000 * i / sampling_frequency for i in range(num_samples)]

        # Create a DataFrame with the signal data, "Sample," and "time_ms"
        ecg_data = pd.DataFrame({'Sample': range(num_samples), 'time_ms': time_ms})
        for i in range(header.n_sig):
            ecg_data[header.sig_name[i]] = record.p_signal[:, i]

        # Read the annotations
        annotations = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')

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
        ecg_data['Annotation'] = ''
        ecg_data['Description'] = ''

        # Add annotations and descriptions to the DataFrame
        for idx, symbol in zip(ann_sample_indices, ann_symbols):
            ecg_data.at[idx, 'Annotation'] = symbol
            ecg_data.at[idx, 'Description'] = annotation_map.get(symbol, 'Unknown')

        # Save the main DataFrame as a CSV file in the output directory
        csv_filename = os.path.join(output_dir, f'{record_name}.csv')
        ecg_data.to_csv(csv_filename, index=False)

        print(f'Record {record_name} saved as {csv_filename}')

        # Prepare content for the text and additional CSV file
        annotation_details = []
        txt_content = []
        txt_content.append(f"Record {record_name} saved as {csv_filename}")
        txt_content.append(f"Annotation samples for {record_name}: {ann_sample_indices[:10]}")
        txt_content.append(f"Annotation symbols for {record_name}: {ann_symbols[:10]}")

        for i in range(len(ann_sample_indices)):
            symbol = ann_symbols[i]
            description = annotation_map.get(symbol, 'Unknown')
            time_stamp = time_ms[ann_sample_indices[i]]
            annotation_info = {
                'Sample index': ann_sample_indices[i],
                'Time (ms)': time_stamp,
                'Symbol': symbol,
                'Description': description,
                'Channels': ', '.join(header.sig_name)
            }
            for sig_name in header.sig_name:
                annotation_info[sig_name] = ecg_data.at[ann_sample_indices[i], sig_name]

            annotation_info_str = f"Sample index: {ann_sample_indices[i]}, Time (ms): {time_stamp:.2f}, Symbol: {symbol}, Description: {description}, Channels: {', '.join(header.sig_name)}, Signal Values: {[annotation_info[sig_name] for sig_name in header.sig_name]}"
            print(annotation_info_str)
            txt_content.append(annotation_info_str)
            annotation_details.append(annotation_info)

        # Check if all desired annotations are present in the file
        present_annotations = set(ann_symbols)
        missing_annotations = set(desired_annotations) - present_annotations

        if missing_annotations:
            missing_info = f"Missing annotations in {record_name}: {missing_annotations}"
            print(missing_info)
            txt_content.append(missing_info)
        else:
            all_present_info = f"All desired annotations are present in {record_name}"
            print(all_present_info)
            txt_content.append(all_present_info)

        # Save annotations to a text file
        txt_filename = os.path.join(output_dir, f'{record_name}.txt')
        with open(txt_filename, 'w') as txt_file:
            for line in txt_content:
                txt_file.write(line + "\n")

        # Save the annotation details to an additional CSV file
        annotation_details_df = pd.DataFrame(annotation_details)
        annotation_csv_filename = os.path.join(output_dir, f'{record_name}_annotations.csv')
        annotation_details_df.to_csv(annotation_csv_filename, index=False)

        # Extract beats and save to a separate CSV file
        beat_segments = extract_beats(record, annotations, signal)

        # Create a DataFrame where each column represents a sample in the beat segment
        max_len = max(len(seg[0]) for seg in beat_segments)
        beat_data = {
            'Symbol': [seg[1] for seg in beat_segments]
        }
        for i in range(max_len):
            beat_data[f'Sample_{i}'] = [seg[0][i] if i < len(seg[0]) else np.nan for seg in beat_segments]

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
