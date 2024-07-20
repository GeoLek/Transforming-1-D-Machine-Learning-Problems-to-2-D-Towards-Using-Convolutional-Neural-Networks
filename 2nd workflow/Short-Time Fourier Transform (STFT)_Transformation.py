import os
import numpy as np
import pandas as pd
from scipy.signal import stft, get_window
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
from skimage import exposure

def load_ecg_data(file_path):
    """Load ECG data from a CSV file."""
    return pd.read_csv(file_path)['MLII'].values

def apply_stft(ecg_data, fs=360, window_function='hann'):
    """Apply Short-Time Fourier Transform (STFT) to ECG data using a specified window function."""
    nperseg = min(256, len(ecg_data))  # Adjusted window size for better frequency resolution
    noverlap = nperseg // 2  # 50% overlap for smoother transitions

    # Use get_window to generate the desired window function
    window = get_window(window_function, nperseg)

    f, t, Zxx = stft(ecg_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, np.abs(Zxx)

def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale using logarithmic scaling."""
    log_scaled_data = np.log1p(image_data)  # Logarithmic scaling to manage the dynamic range effectively
    normalized_data = (log_scaled_data - np.min(log_scaled_data)) / (np.max(log_scaled_data) - np.min(log_scaled_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def adaptive_equalization(image_data):
    """Apply adaptive histogram equalization to the image data for enhanced contrast."""
    img_adapteq = exposure.equalize_adapthist(image_data, clip_limit=0.03)  # CLAHE for better visual contrast
    return (img_adapteq * 255).astype(np.uint8)

def smooth_image(image_data, sigma=2):
    """Apply Gaussian smoothing to the image data for noise reduction."""
    return gaussian_filter(image_data, sigma=sigma)

def create_stft_image(ecg_data, target_shape=(224, 224), max_freq=50, fs=360, window_function='hann'):
    """Create a smoothed grayscale image from ECG data using STFT, with focus on meaningful frequency range."""
    f, stft_data = apply_stft(ecg_data, fs=fs, window_function=window_function)
    freq_idx = f <= max_freq  # Focus on relevant frequencies (up to max_freq Hz)
    stft_data = stft_data[freq_idx, :]
    # Zoom to fit the target shape
    scaled_stft = zoom(stft_data, (target_shape[0] / stft_data.shape[0], target_shape[1] / stft_data.shape[1]), order=1)
    # Convert to grayscale and apply adaptive equalization for contrast enhancement
    grayscale_image = scale_to_grayscale(scaled_stft)
    equalized_image = adaptive_equalization(grayscale_image)
    # Apply Gaussian smoothing for a smoother appearance
    smoothed_image = smooth_image(equalized_image, sigma=2)
    return smoothed_image

def create_and_save_stft_images(record_number, input_file, output_folder, target_shape=(224, 224), window_function='hann'):
    """Process ECG files to create and save STFT-based smoothed grayscale images."""
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
        stft_image = create_stft_image(beat_segment, target_shape, window_function=window_function)
        ecg_pil = Image.fromarray(stft_image)

        # Sanitize the symbol to create a valid file name
        sanitized_symbol = "".join(c if c.isalnum() else '_' for c in symbol)
        # Save the plot as a PNG file with the annotation in the file name
        image_filename = f'beat_{i + 1}_symbol_{sanitized_symbol}.png'
        ecg_pil.save(os.path.join(record_dir, image_filename))

    print(f'Saved images for record {record_number:03d} in {record_dir}')

# Define the directory containing the output CSV files
input_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/output_beats'
output_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Dimension Transformation Images/Short-Time Fourier Transform (STFT)'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of record numbers (e.g., 100, 101, 102, ...)
record_numbers = list(range(100, 235))

# Loop through each record number and process the files
for record_number in record_numbers:
    csv_filename = os.path.join(input_dir, f'{record_number:03d}_beats.csv')
    if os.path.exists(csv_filename):
        create_and_save_stft_images(record_number, csv_filename, output_dir, window_function='hann')  # Change the window function here
    else:
        print(f'CSV file for record {record_number:03d} not found. Skipping...')

print("Processing complete.")
