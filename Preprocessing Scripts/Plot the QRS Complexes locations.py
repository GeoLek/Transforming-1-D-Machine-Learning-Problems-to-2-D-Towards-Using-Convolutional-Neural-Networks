import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_qrs_complexes(qrs_data, lead_name, output_path, record_number):
    # Create a plot for QRS complexes
    plt.figure(figsize=(12, 6))
    plt.scatter(qrs_data['time_ms'], qrs_data['QRS_Amplitude'], color='red', label='QRS Complexes')
    plt.title(f'Detected QRS Complexes - Record {record_number} - {lead_name}')
    plt.xlabel('Time (ms)')
    plt.ylabel('QRS Amplitude')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = f'{record_number}_{lead_name}_QRS_Plot.png'
    plt.savefig(os.path.join(output_path, plot_filename))
    plt.close()
    print(f"Plot saved to {os.path.join(output_path, plot_filename)}")

# Example usage
dataset_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Pan–Tompkins algorithm'
qrs_dataset_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Pan–Tompkins algorithm/QRS complexes'
output_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Pan–Tompkins algorithm/QRS complexes/Visualizations'

# Make sure the output folder exists
os.makedirs(output_path, exist_ok=True)

# Process each CSV and save QRS locations and times
for record_number in range(100, 235):
    ecg_file_path = os.path.join(dataset_path, f'{record_number}.csv')
    if os.path.isfile(ecg_file_path):
        df = pd.read_csv(ecg_file_path)
        lead_columns = [col for col in df.columns if col not in ['Sample', 'time_ms']]

        # Plot QRS complexes for each lead
        for lead in lead_columns:
            qrs_file_path = os.path.join(qrs_dataset_path, f'{record_number}_{lead}_QRS.csv')
            if os.path.isfile(qrs_file_path):
                qrs_data = pd.read_csv(qrs_file_path)
                plot_qrs_complexes(qrs_data, lead, output_path, record_number)
