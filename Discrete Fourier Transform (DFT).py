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

def create_and_save_dft_images(input_folder, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save DFT-based images."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = load_ecg_data(file_path)
            dft_image = create_dft_image(ecg_data, target_shape)
            ecg_pil = Image.fromarray(dft_image)
            image_filename = filename.replace('.csv', '.png')
            ecg_pil.save(os.path.join(output_folder, image_filename))

input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Discrete Fourier Transform (DFT):/2D Images'
create_and_save_dft_images(input_folder, output_folder)
