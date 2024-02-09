import os
import pandas as pd

# Define the input folder where the processed CSV files are located
input_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/Processed CSV files"

# Define the output folder to save the merged CSV files
output_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/MIT-BIH-CSV/Processed CSV files/Combined"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Create a dictionary to store merged data frames for each record
merged_data_dict = {}

# Iterate through record numbers from 100 to 234
for record_id in range(100, 235):
    record_id_str = str(record_id)
    merged_data = None

    # Look for all CSV files that start with the record number
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.startswith(f"{record_id_str}.csv_"):
                file_path = os.path.join(root, file)

                try:
                    # Attempt to read the CSV file
                    df = pd.read_csv(file_path)

                    # Rename columns to include the lead name (e.g., MLII or V5)
                    lead_name = file.split(".csv_")[1].split("_processed.csv")[0]
                    df.rename(columns={"time_ms": "time_ms", lead_name: lead_name}, inplace=True)

                    # Merge the data frames
                    if merged_data is None:
                        merged_data = df
                    else:
                        merged_data = pd.merge(merged_data, df, on="time_ms", how="outer")
                except UnicodeDecodeError:
                    print(f"Error reading file: {file_path}. Skipping.")

    # Save the merged data to a CSV file for the record
    if merged_data is not None:
        output_file = os.path.join(output_folder, f"{record_id_str}_merged_data.csv")
        merged_data.sort_values(by="time_ms", inplace=True)
        merged_data.to_csv(output_file, index=False)
        print(f"Saved merged data for record {record_id_str} to {output_file}.")