import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory containing the output CSV files
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Function to sanitize file names
def sanitize_filename(filename):
    return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in filename).strip()

# Function to plot and save beat segments from a CSV file
def plot_and_save_beats(record_number, csv_filename):
    # Load the CSV file
    beats_df = pd.read_csv(csv_filename)

    # Extract symbols and beat segments
    symbols = beats_df['Symbol'].tolist()
    beat_segments = beats_df.drop(columns=['Symbol']).values

    # Create a directory for the record
    record_dir = os.path.join(output_dir, f'record_{record_number:03d}')
    os.makedirs(record_dir, exist_ok=True)

    # Plot and save each beat segment as a PNG file
    for i in range(len(beat_segments)):
        plt.figure(figsize=(10, 4))
        plt.plot(beat_segments[i])
        plt.title(f'Record {record_number:03d} - Beat {i + 1} - Symbol: {symbols[i]}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = sanitize_filename(symbols[i] if symbols[i] else 'Unknown')
        # Save the plot as a PNG file with the annotation in the file name
        png_filename = os.path.join(record_dir, f'beat_{i + 1}_symbol_{sanitized_symbol}.png')
        plt.savefig(png_filename)
        plt.close()

    print(f'Saved beats for record {record_number:03d} in {record_dir}')

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(output_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        plot_and_save_beats(record_number, csv_filename)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")