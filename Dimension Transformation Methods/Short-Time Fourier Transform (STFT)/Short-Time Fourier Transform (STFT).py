import os
import numpy as np
import pandas as pd
from scipy.signal import stft, hann
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image
from skimage import exposure

def load_ecg_data(file_path):
    """Load ECG data from a CSV file."""
    return pd.read_csv(file_path)['MLII'].values

def apply_stft(ecg_data, fs=360):
    """Apply Short-Time Fourier Transform (STFT) to ECG data using a Hann window."""
    nperseg = min(256, len(ecg_data))  # Adjusted window size for better frequency resolution
    noverlap = nperseg // 2  # 50% overlap for smoother transitions
    window = hann(nperseg)  # Hann window to reduce spectral leakage
    f, t, Zxx = stft(ecg_data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, np.abs(Zxx)

def scale_to_grayscale(image_data):
    """Normalize the image data and convert it to grayscale using logarithmic scaling."""
    # Logarithmic scaling to manage the dynamic range effectively
    log_scaled_data = np.log1p(image_data)
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

def create_stft_image(ecg_data, target_shape=(224, 224), max_freq=50, fs=360):
    """Create a smoothed grayscale image from ECG data using STFT, with focus on meaningful frequency range."""
    f, stft_data = apply_stft(ecg_data, fs=fs)
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

def create_and_save_stft_images(input_folder, output_folder, target_shape=(224, 224)):
    """Process ECG files to create and save STFT-based smoothed grayscale images."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.startswith('normal_') and filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = load_ecg_data(file_path)
            stft_image = create_stft_image(ecg_data, target_shape)
            ecg_pil = Image.fromarray(stft_image)
            image_filename = filename.replace('.csv', '.png')
            ecg_pil.save(os.path.join(output_folder, image_filename))


input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)//2D Images'
create_and_save_stft_images(input_folder, output_folder)