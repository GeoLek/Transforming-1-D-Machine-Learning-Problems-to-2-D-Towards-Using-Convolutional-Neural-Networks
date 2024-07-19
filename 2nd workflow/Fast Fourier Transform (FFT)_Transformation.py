import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom
from PIL import Image
from numpy.fft import fft, fftshift

def scale_to_grayscale(image_data):
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def create_fft_image(ecg_data, target_shape=(224, 224)):
    fft_data = fft(ecg_data)
    fft_data_shifted = fftshift(fft_data)
    magnitude_spectrum = np.abs(fft_data_shifted)
    height_scale = target_shape[0] / magnitude_spectrum.shape[0]
    scaled_spectrum = zoom(magnitude_spectrum, height_scale, order=1)
    scaled_spectrum_2d = np.tile(scaled_spectrum, (target_shape[1], 1)).T
    grayscale_image = scale_to_grayscale(scaled_spectrum_2d)
    return grayscale_image

def create_and_save_fft_images(record_number, input_file, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save FFT-based images."""
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
        fft_image = create_fft_image(beat_segment, target_shape)
        ecg_pil = Image.fromarray(fft_image)

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        ecg_pil.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_images_fft'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_fft_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
