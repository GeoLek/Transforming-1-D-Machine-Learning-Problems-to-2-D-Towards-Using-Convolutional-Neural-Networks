import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom
from PIL import Image

def scale_to_grayscale(ecg_data):
    normalized_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def create_and_save_images(record_number, input_file, output_folder, target_shape=(224, 224)):
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
        ecg_grayscale = scale_to_grayscale(beat_segment)

        # Calculate the stretch factor for height
        stretch_factor_height = target_shape[0] / len(ecg_grayscale)

        # Scale the ECG signal to fit the height of the target image
        stretched_signal_height = zoom(ecg_grayscale, stretch_factor_height, order=1)

        # Calculate the width based on the target shape
        signal_height = stretched_signal_height.shape[0]
        signal_width = int(len(ecg_grayscale) * (target_shape[1] / target_shape[0]))

        # Scale the ECG signal to fit the width of the target image
        stretched_signal_width = zoom(ecg_grayscale, signal_width / len(ecg_grayscale), order=1)

        # Create a 2D image with the signal in the center
        ecg_image = np.zeros(target_shape, dtype=np.uint8)
        start_row = (target_shape[0] - signal_height) // 2
        end_row = start_row + signal_height
        start_col = (target_shape[1] - signal_width) // 2
        end_col = start_col + signal_width
        ecg_image[start_row:end_row, start_col:end_col] = stretched_signal_width

        # Convert to PIL Image and resize to 224x224 pixels
        ecg_pil = Image.fromarray(ecg_image)
        ecg_pil = ecg_pil.resize(target_shape, Image.BILINEAR)

        # Save the resized image with the annotation in the file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        ecg_pil.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
