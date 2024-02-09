import pandas as pd
import numpy as np
import os
from scipy.ndimage import zoom
from PIL import Image
from numpy.fft import fft, fftshift

def scale_to_grayscale_with_clipping(image_data, clip_percentile=99):
    vmax = np.percentile(image_data, clip_percentile)
    vmin = np.percentile(image_data, 100 - clip_percentile)
    clipped_data = np.clip(image_data, vmin, vmax)
    normalized_data = (clipped_data - vmin) / (vmax - vmin)
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def create_fft_image(ecg_data, target_shape=(224, 224)):
    fft_data = fft(ecg_data)
    fft_data_shifted = fftshift(fft_data)
    magnitude_spectrum = np.abs(fft_data_shifted)
    height_scale = target_shape[0] / magnitude_spectrum.shape[0]
    scaled_spectrum = zoom(magnitude_spectrum, height_scale, order=1)
    scaled_spectrum_2d = np.tile(scaled_spectrum, (target_shape[1], 1)).T
    grayscale_image = scale_to_grayscale_with_clipping(scaled_spectrum_2d)
    return grayscale_image

def create_and_save_fft_images(input_folder, output_folder, target_shape=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.startswith('normal_') and filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = pd.read_csv(file_path)['MLII'].values
            fft_image = create_fft_image(ecg_data, target_shape)
            ecg_pil = Image.fromarray(fft_image)
            image_filename = filename.replace('.csv', '.png')
            ecg_pil.save(os.path.join(output_folder, image_filename))

input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Fast Fourier Transform (FFT)/FFT with Linear Scaling and Clipping/2D Images'
create_and_save_fft_images(input_folder, output_folder)
