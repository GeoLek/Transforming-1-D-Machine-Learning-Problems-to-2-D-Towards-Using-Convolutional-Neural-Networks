import os
import shutil

# Define the main directory containing the transformation method folders
main_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Dimension Transformation Images/2D visualization'

# Define the new base directory where the organized files will be copied
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Organized_Beats/2D visualization'

# Define the annotations
annotations = ['N', 'S', 'V', 'F', 'Q']

# Ensure the output directories for each annotation exist
for annotation in annotations:
    os.makedirs(os.path.join(output_base_dir, annotation), exist_ok=True)

# Loop through each record folder
for record_folder in os.listdir(main_dir):
    record_path = os.path.join(main_dir, record_folder)

    # Check if it's a directory and matches the pattern for record folders
    if os.path.isdir(record_path) and record_folder.startswith('record_'):
        record_number = record_folder.split('_')[1]  # Extract the record number

        # Loop through each file in the record folder
        for filename in os.listdir(record_path):
            if filename.endswith('.png'):
                # Extract the annotation symbol from the filename
                parts = filename.split('_')
                if len(parts) >= 4 and parts[2] == 'symbol':
                    annotation_symbol = parts[3].split('.')[0]

                    # Check if the annotation symbol is in the desired list
                    if annotation_symbol in annotations:
                        # Create a unique filename by including the record number
                        unique_filename = f'record_{record_number}_{filename}'

                        # Define the source and destination file paths
                        src_file = os.path.join(record_path, filename)
                        dst_file = os.path.join(output_base_dir, annotation_symbol, unique_filename)

                        # Copy the file to the corresponding annotation directory
                        shutil.copy(src_file, dst_file)
                        print(f'Copied {src_file} to {dst_file}')  # Print statement for debugging

print("Organizing complete.")
