import os
import random
import shutil

# Define your dataset directory and the output directories for splits
dataset_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method/2D Images'
output_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method'

# Create output folders for train, validation, and test
train_dir = os.path.join(output_dir, 'train')
validation_dir = os.path.join(output_dir, 'validation')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List of all files in the dataset directory
file_list = [f'paced_{i}.png' for i in range(2137)]  # Adjust the range based on your dataset size

# Shuffle the file list
random.shuffle(file_list)

# Split the data into train, validation, and test
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

total_files = len(file_list)
train_split = int(train_ratio * total_files)
validation_split = int(validation_ratio * total_files)

# Move files to the respective directories
for i, file_name in enumerate(file_list):
    src_path = os.path.join(dataset_dir, file_name)
    if i < train_split:
        dst_dir = train_dir
    elif i < train_split + validation_split:
        dst_dir = validation_dir
    else:
        dst_dir = test_dir
    dst_path = os.path.join(dst_dir, file_name)

    try:
        shutil.move(src_path, dst_path)
    except FileNotFoundError:
        print(f"Warning: {file_name} not found, skipping.")

print(
    f"Dataset split into {train_ratio * 100}% train, {validation_ratio * 100}% validation, and {test_ratio * 100}% test.")