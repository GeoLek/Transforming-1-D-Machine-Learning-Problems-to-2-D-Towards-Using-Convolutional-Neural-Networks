import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
from PIL import Image

def scale_to_grayscale(ecg_data):
    normalized_data = (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))
    grayscale_data = (normalized_data * 255).astype(np.uint8)
    return grayscale_data

def create_and_save_images(input_folder, output_folder, target_shape=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.startswith('normal_') and filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = pd.read_csv(file_path)['MLII'].values
            ecg_grayscale = scale_to_grayscale(ecg_data)

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

            # Save the resized image
            image_filename = filename.replace('.csv', '.png')
            ecg_pil.save(os.path.join(output_folder, image_filename))

input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method'
create_and_save_images(input_folder, output_folder)
