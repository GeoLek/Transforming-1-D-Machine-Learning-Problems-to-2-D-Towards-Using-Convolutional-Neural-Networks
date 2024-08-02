import os

def rename_files_in_annotation_folders(base_folder, annotations):
    for annotation in annotations:
        folder_path = os.path.join(base_folder, annotation)
        # Get a list of files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Sort the files to maintain the order
        files.sort()

        # Iterate over the files and rename them
        for i, filename in enumerate(files):
            # Define the new file name
            new_filename = f"{annotation}_{i}.png"

            # Full path for old and new file names
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file, new_file)

            # Optional: Print the renaming info
            print(f"Renamed {old_file} to {new_file}")

# Example usage
base_folder = ('/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Augmented_Beats/Short-Time Fourier Transform (STFT)')  # Replace with your base folder path
annotations = ['N', 'S', 'V', 'F', 'Q']  # List of annotation folders

rename_files_in_annotation_folders(base_folder, annotations)

print("Renaming complete.")
