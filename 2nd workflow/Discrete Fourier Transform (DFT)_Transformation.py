import os
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from PIL import Image

def load_ecg_data(file_path):
    """Load ECG data from a CSV file."""
    return pd.read_csv(file_path)['MLII'].values

def apply_dft(ecg_data):
    """Apply Discrete Fourier Transform (DFT) to the ECG data."""
    dft_data = np.fft.fft(ecg_data)
    dft_shifted = np.fft.fftshift(dft_data)  # Shift the zero frequency component to the center
    magnitude_spectrum = np.abs(dft_shifted)
    return magnitude_spectrum

def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale."""
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def create_dft_image(ecg_data, target_shape=(224, 224)):
    """Create a DFT-based grayscale image from ECG data."""
    dft_data = apply_dft(ecg_data)
    # Assuming DFT data is 1D, convert it to 2D by repeating it to match the target shape
    dft_image_2d = np.tile(dft_data, (target_shape[0], 1))
    # Scale the 2D image to the target size
    scaled_image = zoom(dft_image_2d, (1, target_shape[1] / dft_image_2d.shape[1]), order=1)
    return scale_to_grayscale(scaled_image)

def create_and_save_dft_images(record_number, input_file, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save DFT-based images."""
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
        dft_image = create_dft_image(beat_segment, target_shape)
        ecg_pil = Image.fromarray(dft_image)

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        ecg_pil.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_images_dft'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_dft_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
