import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

# Define global parameters for easy modification
EPS = 0.05  # Proximity threshold for recurrences
STEPS = 1   # Steps for considering recurrences

def load_ecg_data(file_path):
    return pd.read_csv(file_path)['MLII'].values

def create_recurrence_plot(data, eps=EPS, steps=STEPS):
    n = len(data)
    recurrence_matrix = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i - steps, i + steps + 1):
            if 0 <= j < n:
                if abs(data[i] - data[j]) < eps:
                    recurrence_matrix[i, j] = 1
    return recurrence_matrix

def create_and_save_rp_images(record_number, input_file, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save Recurrence Plot-based images."""
    os.makedirs(output_folder, exist_ok=True)

    # Load the beat segments from the CSV file
    beats_df = pd.read_csv(input_file)
    symbols = beats_df['Symbol'].tolist()
    beat_segments = beats_df.drop(columns=['Symbol']).values

    # Create output directory for the record
    record_dir = os.path.join(output_folder, f'record_{record_number:03d}')
    os.makedirs(record_dir, exist_ok=True)

    # Process and save each beat segment as an image
    for i, (beat_segment, symbol) in enumerate(zip(beat_segments, symbols)):
        rp_matrix = create_recurrence_plot(beat_segment)
        resized_rp = zoom(rp_matrix, (target_shape[0] / rp_matrix.shape[0], target_shape[1] / rp_matrix.shape[1]), order=0)
        rp_image = Image.fromarray(resized_rp * 255)  # Convert binary matrix to an image

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        rp_image.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Dimension Transformation Images/Recurrence Plots'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_rp_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
