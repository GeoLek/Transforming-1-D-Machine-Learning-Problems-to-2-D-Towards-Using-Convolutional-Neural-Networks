import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the CSV files and the output folder
folder_path = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/107_P"  # Update this to your folder path
output_folder = "/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Segmentation results/Visualizations/107"  # Update this to your output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through files in the folder
file_count = 0
while True:
    file_name = f"107_segment_{file_count}.csv"  # Adjust the prefix to match your naming convention
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        break

    # Reading ECG Data into a DataFrame
    ecg = pd.read_csv(file_path)

    # Extract the first value of "time_ms" and convert it to minutes
    time_in_minutes = ecg["time_ms"].iloc[0] / 60000  # 60000 ms in a minute

    # Plot the Entire 30-Minute EKG for MLII
    plt.figure(figsize=(12, 4))
    plt.plot(ecg["time_ms"], ecg["MLII"], label="MLII", color="blue")
    plt.title(f"30-minute EKG of Patient 100 (MLII) - Segment {file_count}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Save the plot in the output folder with the time in minutes as part of the name
    plot_filename = os.path.join(output_folder, f"{time_in_minutes:.2f}_minutes_plot_{file_count}.png")
    plt.savefig(plot_filename)
    plt.close()

    # Display the plot
    plt.show()

    file_count += 1

print(f"Plots saved in folder: {output_folder}")