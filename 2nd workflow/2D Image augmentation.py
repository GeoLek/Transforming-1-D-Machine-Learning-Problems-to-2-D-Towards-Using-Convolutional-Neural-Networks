import os
import shutil
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Organized_Beats/Short-Time Fourier Transform (STFT)'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Augmented_Beats/Short-Time Fourier Transform (STFT)'

# Define the target number of samples per class
target_samples = 100000

# Ensure the output directories for each annotation exist
annotations = ['N', 'S', 'V', 'F', 'Q']
for annotation in annotations:
    os.makedirs(os.path.join(output_base_dir, annotation), exist_ok=True)

# Augmentation functions
def light_rotation(image):
    return image.rotate(random.uniform(-5, 5))

def slight_zoom(image):
    scale_factor = random.uniform(0.95, 1.05)
    width, height = image.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    return image.crop((left, top, right, bottom))

def slight_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.95, 1.05))

def slight_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(0.95, 1.05))

def slight_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 0.01, np_image.shape)
    noisy_image = np.clip(np_image + noise * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def slight_blur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))

def augment_image(image):
    augmentation_methods = [light_rotation, slight_zoom, slight_brightness, slight_contrast, slight_noise, slight_blur]
    augmented_image = image
    num_methods = random.randint(1, len(augmentation_methods))
    methods = random.sample(augmentation_methods, num_methods)
    for method in methods:
        augmented_image = method(augmented_image)
    return augmented_image

# Process each annotation folder
for annotation in annotations:
    input_dir = os.path.join(input_base_dir, annotation)
    output_dir = os.path.join(output_base_dir, annotation)

    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    num_files = len(files)
    print(f"Processing {annotation}: {num_files} files found.")

    # Copy original files to the output directory
    for file in files:
        src_file = os.path.join(input_dir, file)
        dst_file = os.path.join(output_dir, file)
        shutil.copy(src_file, dst_file)

    # Perform data augmentation to reach the target number of samples
    additional_samples_needed = target_samples - num_files
    if additional_samples_needed > 0:
        print(f"Augmenting {annotation}: Need {additional_samples_needed} more samples.")
        for i in range(additional_samples_needed):
            if i % 1000 == 0:  # Print status every 1000 samples
                print(f"Augmenting {annotation}: {additional_samples_needed - i} samples remaining.")
            original_file = random.choice(files)
            src_file = os.path.join(input_dir, original_file)
            with Image.open(src_file) as img:
                augmented_img = augment_image(img)
                augmented_filename = f"augmented_{i+1}_{original_file}"
                augmented_img.save(os.path.join(output_dir, augmented_filename))
    else:
        print(f"No augmentation needed for {annotation}.")

print("Data augmentation complete.")
