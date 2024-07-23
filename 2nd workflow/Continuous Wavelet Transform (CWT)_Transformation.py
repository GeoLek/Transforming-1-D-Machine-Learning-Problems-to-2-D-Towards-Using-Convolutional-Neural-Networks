import os
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
from skimage import exposure

# Define global parameters for easy modification
WAVELET_NAME = 'morl'  # Options: 'morl', 'gaus8', 'gaus4', etc.
SIGMA = 0.1            # Gaussian smoothing parameter
CLIP_LIMIT = 0.03      # Contrast enhancement parameter

def load_ecg_data(file_path):
    return pd.read_csv(file_path)['MLII'].values

def apply_cwt(ecg_data, scales, wavelet_name=WAVELET_NAME):
    coefficients, frequencies = pywt.cwt(ecg_data, scales, wavelet_name)
    return np.abs(coefficients)

def scale_to_grayscale(image_data):
    normalized_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def enhance_contrast(image_data, clip_limit=CLIP_LIMIT):
    img_eq = exposure.equalize_adapthist(image_data, clip_limit=clip_limit)
    return (img_eq * 255).astype(np.uint8)

def smooth_image(image_data, sigma=SIGMA):
    return gaussian_filter(image_data, sigma=sigma)

def create_cwt_image(ecg_data, target_shape=(224, 224)):
    scales = np.arange(1, 128)
    cwt_magnitude = apply_cwt(ecg_data, scales)
    resized_cwt = zoom(cwt_magnitude, (target_shape[0] / cwt_magnitude.shape[0], target_shape[1] / cwt_magnitude.shape[1]), order=1)
    grayscale_image = scale_to_grayscale(resized_cwt)
    contrast_enhanced_image = enhance_contrast(grayscale_image)
    return smooth_image(contrast_enhanced_image)

def create_and_save_cwt_images(record_number, input_file, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save CWT-based images."""
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
        cwt_image = create_cwt_image(beat_segment, target_shape)
        ecg_pil = Image.fromarray(cwt_image)

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        ecg_pil.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Dimension Transformation Images/Continuous Wavelet Transform (CWT)'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_cwt_images(record_number, csv_filename, output_dir)
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
