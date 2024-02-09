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

def create_and_save_fft_images(input_folder, output_folder, target_shape=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    file_count = 0

    for filename in os.listdir(input_folder):
        if filename.startswith('normal_') and filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            print(f"Reading file: {file_path}")
            ecg_data = pd.read_csv(file_path)['MLII'].values

            if ecg_data.size == 0:
                print(f"Warning: No data in {filename}. Skipping.")
                continue

            fft_image = create_fft_image(ecg_data, target_shape)

            ecg_pil = Image.fromarray(fft_image)
            image_filename = filename.replace('.csv', '.png')
            save_path = os.path.join(output_folder, image_filename)
            ecg_pil.save(save_path)

            print(f"Saved FFT image to {save_path}")
            file_count += 1

    print(f"Total files processed: {file_count}")

input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Fast Fourier Transform (FFT)/Pure FFT/2D Images'
create_and_save_fft_images(input_folder, output_folder)
