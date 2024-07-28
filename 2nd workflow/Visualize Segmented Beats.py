import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_segmented_beats(segmented_beats_file, output_folder):
    # Load the segmented beats data
    segmented_data = pd.read_csv(segmented_beats_file)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract the time data assuming the time is for each sample in the row
    sample_indices = segmented_data['Sample index'].values
    time_data = segmented_data['Time (ms)'].values

    # Extract signal data columns (skip the first columns which are metadata)
    signal_columns = segmented_data.columns[5:]  # Assuming signal columns start from the 6th column

    # Iterate over each segment and plot the beat
    for index, row in segmented_data.iterrows():
        symbol = row['Symbol']

        # Extract signal data for each lead
        for signal_column in signal_columns:
            signal_data = row[signal_column]

            # Check if signal_data is iterable (list-like)
            if isinstance(signal_data, str):
                try:
                    signal_data = eval(signal_data)  # Convert string representation of list to actual list
                except Exception as e:
                    print(f"Error evaluating signal data for {signal_column} in row {index}: {e}")
                    continue  # Skip if conversion fails

            # Ensure signal data is a numpy array
            if isinstance(signal_data, (list, np.ndarray)):
                signal_data = np.array(signal_data)
            else:
                print(f"Skipping non-list-like signal data for {signal_column} in row {index}")
                continue  # Skip if signal_data is not list-like

            # Create a time axis for the signal data
            segment_length = len(signal_data)
            time_axis = np.linspace(0, segment_length, segment_length)

            # Plot the beat segment
            plt.figure(figsize=(10, 4))
            plt.plot(time_axis, signal_data, label=f'ECG Signal ({signal_column})')
            plt.title(f"Beat Segment {index + 1} - Symbol: {symbol}")
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.legend()

            # Create a directory for the annotation symbol if it doesn't exist
            annotation_dir = os.path.join(output_folder, symbol)
            os.makedirs(annotation_dir, exist_ok=True)

            # Save the plot
            plot_file_name = f"{os.path.splitext(os.path.basename(segmented_beats_file))[0]}_segment_{index + 1}_{signal_column}.png"
            plt.savefig(os.path.join(annotation_dir, plot_file_name))
            plt.close()

            print(f"Saved plot for {signal_column} in segment {index + 1} with symbol {symbol}")


# Parameters
segmented_beats_folder = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Segmented_Beats'
output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Visualize_Segmented_Beats'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Process each CSV file and visualize the segmented beats
for filename in os.listdir(segmented_beats_folder):
    if filename.endswith('.csv'):
        segmented_file_path = os.path.join(segmented_beats_folder, filename)

        print(f"Processing segmented beats file: {segmented_file_path}")
        visualize_segmented_beats(segmented_file_path, output_path)

print("Visualization of segmented beats complete.")