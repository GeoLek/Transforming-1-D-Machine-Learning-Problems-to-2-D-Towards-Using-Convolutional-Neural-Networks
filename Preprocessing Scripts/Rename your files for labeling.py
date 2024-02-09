import os

def rename_files_in_folder(folder_path, new_name_prefix):
    # Get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort the files to maintain the order
    files.sort()

    # Iterate over the files and rename them
    for i, filename in enumerate(files):
        # Define the new file name
        new_filename = f"{new_name_prefix}_{i}.csv"

        # Full path for old and new file names
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)

        # Optional: Print the renaming info
        print(f"Renamed {old_file} to {new_file}")

# Example usage
folder_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Paced beats'  # Replace with your folder path
new_name_prefix = 'paced'  # The new name prefix for the files
rename_files_in_folder(folder_path, new_name_prefix)
