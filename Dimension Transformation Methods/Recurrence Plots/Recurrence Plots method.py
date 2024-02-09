import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

# Define global parameters for easy modification
EPS = 0.05  # Proximity threshold for recurrences
STEPS = 1   # Steps for considering recurrences

def load_ecg_data(file_path):
    return pd.read_csv(file_path)['MLII'].values

def create_recurrence_plot(data, eps=EPS, steps=STEPS):
    n = len(data)
    recurrence_matrix = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i - steps, i + steps + 1):
            if 0 <= j < n:
                if abs(data[i] - data[j]) < eps:
                    recurrence_matrix[i, j] = 1
    return recurrence_matrix

def create_and_save_rp_images(input_folder, output_folder, target_shape=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            ecg_data = load_ecg_data(file_path)
            rp_matrix = create_recurrence_plot(ecg_data)
            resized_rp = zoom(rp_matrix, (target_shape[0] / rp_matrix.shape[0], target_shape[1] / rp_matrix.shape[1]), order=0)
            rp_image = Image.fromarray(resized_rp * 255)  # Convert binary matrix to an image
            image_filename = filename.replace('.csv', '.png')
            rp_image.save(os.path.join(output_folder, image_filename))

# Specify your actual input and output folder paths
input_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Normal beats'
output_folder = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Recurrence Plots/2D Images'
create_and_save_rp_images(input_folder, output_folder)
