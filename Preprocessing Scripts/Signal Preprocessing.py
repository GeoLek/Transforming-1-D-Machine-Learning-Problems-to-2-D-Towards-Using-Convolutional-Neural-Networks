import pandas as pd
import numpy as np
from scipy.signal import medfilt, detrend
import os

def filter_noise(ecg_signal, kernel_size=3):
    return medfilt(ecg_signal, kernel_size=kernel_size)

def remove_baseline_wander(ecg_signal):
    return detrend(ecg_signal, type='linear')

def normalize_signal(ecg_signal):
    return (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

def preprocess_ecg_signal(ecg_data):
    # Preprocess each lead in the DataFrame
    for column in ecg_data.columns[2:]:  # Skipping 'sample' and 'time_ms' columns
        ecg_data[column] = filter_noise(ecg_data[column])
        ecg_data[column] = remove_baseline_wander(ecg_data[column])
        ecg_data[column] = normalize_signal(ecg_data[column])
    return ecg_data

def preprocess_and_save_ecg(dataset_path, output_path):
    for record_number in range(100, 235):
        file_name = f'{record_number}.csv'
        file_path = os.path.join(dataset_path, file_name)

        if os.path.exists(file_path):
            try:
                ecg_data = pd.read_csv(file_path)
                ecg_processed = preprocess_ecg_signal(ecg_data)

                output_file_name = os.path.join(output_path, file_name)
                ecg_processed.to_csv(output_file_name, index=False)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"File {file_path} not found. Skipping.")


# Example usage
dataset_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-data/CSV'  # Replace with your actual dataset folder path
output_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/Processed CSV files'    # Replace with your desired output folder path
os.makedirs(output_path, exist_ok=True)
preprocess_and_save_ecg(dataset_path, output_path)