import os
import shutil
import random

# Define the input and output directories
input_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Augmented_Beats/2D visualization'
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Data/2D visualization'
splits = ['train', 'val', 'test']
annotations = ['N', 'S', 'V', 'F', 'Q']

# Define the split ratios
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Ensure the output directories for each annotation and split exist
for split in splits:
    for annotation in annotations:
        os.makedirs(os.path.join(output_base_dir, split, annotation), exist_ok=True)


# Function to split files into train, validation, and test sets
def split_files(files, ratios):
    random.shuffle(files)
    total = len(files)
    train_end = int(ratios['train'] * total)
    val_end = train_end + int(ratios['val'] * total)
    return {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }


# Process each annotation folder
for annotation in annotations:
    input_dir = os.path.join(input_base_dir, annotation)
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Split the files
    split_files_dict = split_files(files, split_ratios)

    # Move the files to the corresponding split directory
    for split in splits:
        for file in split_files_dict[split]:
            src_file = os.path.join(input_dir, file)
            dst_file = os.path.join(output_base_dir, split, annotation, file)
            shutil.copy(src_file, dst_file)

    print(f"Processed {annotation}: {len(files)} files split into train, val, and test.")

print("Data splitting complete.")
