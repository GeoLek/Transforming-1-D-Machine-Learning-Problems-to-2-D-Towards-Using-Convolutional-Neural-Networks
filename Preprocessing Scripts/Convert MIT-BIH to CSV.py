import os
import pandas as pd
import wfdb

# Directory containing the MIT-BIH data files
data_dir = '/your path here/MIT-data' #The folder with all the records (e.g 100.hea,100.atr ... 101.dat, 101.hea etc)

# Output directory where you want to save the CSV files
output_dir = '/your path here/MIT-BIH-CSV'  # Specify your output directory

# Ensure the output directory exists; create it if necessary
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    record_name = f'{record_number:03d}'  # Format record number with leading zeros

    try:
        # Load the header file to get channel names
        header = wfdb.rdheader(os.path.join(data_dir, record_name))

        # Load the ECG signal data
        record = wfdb.rdrecord(os.path.join(data_dir, record_name))

        # Calculate the "time_ms" column based on the sampling frequency
        sampling_frequency = header.fs  # Sampling frequency in Hz
        num_samples = len(record.p_signal)
        time_ms = [1000 * i / sampling_frequency for i in range(num_samples)]

        # Create a DataFrame with the signal data, "Sample," and "time_ms"
        ecg_data = pd.DataFrame({'Sample': range(num_samples), 'time_ms': time_ms})
        for i in range(header.n_sig):
            ecg_data[header.sig_name[i]] = record.p_signal[:, i]

        # Save the DataFrame as a CSV file in the output directory
        csv_filename = os.path.join(output_dir, f'{record_name}.csv')
        ecg_data.to_csv(csv_filename, index=False)

        print(f'Record {record_name} saved as {csv_filename}')
    except FileNotFoundError:
        print(f'Record {record_name} not found. Skipping...')

print("Processing complete.")