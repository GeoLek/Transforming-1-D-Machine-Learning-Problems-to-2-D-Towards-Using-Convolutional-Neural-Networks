import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_segmented_beats(segmented_beats_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all CSV files in the segmented beats folder
    csv_files = [file for file in os.listdir(segmented_beats_folder) if file.endswith('.csv')]

    # Process each CSV file and visualize the segmented beats
    for filename in csv_files:
        segmented_file_path = os.path.join(segmented_beats_folder, filename)
        print(f"Processing segmented beats file: {segmented_file_path}")

        # Load the segmented beats data
        segmented_data = pd.read_csv(segmented_file_path)

        # Extract signal data columns (skip the first columns which are metadata)
        signal_columns = segmented_data.columns[5:]  # Assuming signal columns start from the 6th column

        # Iterate over each segment and plot the beat
        for index, row in segmented_data.iterrows():
            symbol = row['Symbol']

            # Extract signal data for each lead
            for signal_column in signal_columns:
                signal_data = row[signal_column]

                # Convert signal_data to numpy array
                if isinstance(signal_data, str):
                    try:
                        signal_data = np.fromstring(signal_data.strip('[]'), sep=',')
                    except ValueError as e:
                        print(f"Error converting signal data for {signal_column} in row {index}: {e}")
                        continue
                elif isinstance(signal_data, (list, np.ndarray)):
                    signal_data = np.array(signal_data)
                else:
                    print(f"Skipping non-list-like signal data for {signal_column} in row {index}")
                    continue

                # Ensure valid signal data
                if len(signal_data) == 0:
                    print(f"Skipping empty signal data for {signal_column} in row {index}")
                    continue

                # Create a time axis for the signal data
                time_data = row['Time (ms)']

                # Convert time_data to numpy array
                if isinstance(time_data, str):
                    try:
                        time_data = np.fromstring(time_data.strip('[]'), sep=',')
                    except ValueError as e:
                        print(f"Error converting time data for {signal_column} in row {index}: {e}")
                        continue
                elif isinstance(time_data, (list, np.ndarray)):
                    time_data = np.array(time_data)
                else:
                    print(f"Skipping non-list-like time data for {signal_column} in row {index}")
                    continue

                # Ensure valid time data
                if len(time_data) == 0:
                    print(f"Skipping empty time data for {signal_column} in row {index}")
                    continue

                # Plot the beat segment
                plt.figure(figsize=(10, 4))
                plt.plot(time_data, signal_data, label=f'ECG Signal ({signal_column})')
                plt.title(f"Beat Segment {index + 1} - Symbol: {symbol}")
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude')
                plt.legend()

                # Create a directory for the annotation symbol if it doesn't exist
                annotation_dir = os.path.join(output_folder, symbol)
                os.makedirs(annotation_dir, exist_ok=True)

                # Save the plot
                plot_file_name = f"{os.path.splitext(os.path.basename(segmented_file_path))[0]}_segment_{index + 1}_{signal_column}.png"
                plt.savefig(os.path.join(annotation_dir, plot_file_name))
                plt.close()

                print(f"Saved plot for {signal_column} in segment {index + 1} with symbol {symbol}")


# Parameters
segmented_beats_folder = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Segmented_Beats'
output_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Visualize_Segmented_Beats'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Visualize segmented beats
visualize_segmented_beats(segmented_beats_folder, output_path)

print("Visualization of segmented beats complete.")
