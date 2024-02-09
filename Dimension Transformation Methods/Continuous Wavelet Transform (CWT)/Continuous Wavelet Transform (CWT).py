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

def create_and_save_cwt_images(input_folder, output_folder, target_shape=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = load_ecg_data(file_path)
            cwt_image = create_cwt_image(ecg_data, target_shape)
            ecg_pil = Image.fromarray(cwt_image)
            image_filename = f"{filename.replace('.csv', '')}.png"
            ecg_pil.save(os.path.join(output_folder, image_filename))



# Specify your actual input and output folder paths
input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/2D Images'
create_and_save_cwt_images(input_folder, output_folder)
