import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_signal_comparison(edf_data, csv_data, channel_names, sampling_rate, sample_segment=1000):
    """Plot a segment of signals from both EDF and CSV to compare."""
    time = np.linspace(0, sample_segment / sampling_rate, sample_segment)
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(10, 8))
    for i, channel in enumerate(channel_names):
        edf_segment = edf_data[channel][:sample_segment]
        csv_segment = csv_data[channel].values[:sample_segment]

        axes[i].plot(time, edf_segment, label="EDF", alpha=0.7)
        axes[i].plot(time, csv_segment, label="CSV", linestyle='--', alpha=0.7)
        axes[i].set_title(f"Channel: {channel}")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
        axes[i].legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def check_sampling_rate_duration(edf_info, csv_data):
    """Check if the sampling rate and data length are consistent."""
    edf_duration = edf_info['n_samples'] / edf_info['sfreq']
    csv_duration = len(csv_data) / edf_info['sfreq']

    print(f"EDF Duration: {edf_duration:.2f} s")
    print(f"CSV Duration: {csv_duration:.2f} s")
    assert np.isclose(edf_duration, csv_duration, atol=1), "Duration mismatch between EDF and CSV."


def check_channel_names(edf_channel_names, csv_channel_names):
    """Ensure the channel names are consistent between EDF and CSV."""
    print("EDF Channels:", edf_channel_names)
    print("CSV Channels:", csv_channel_names)
    assert edf_channel_names == csv_channel_names, "Channel names mismatch."


def check_signal_baseline_offset(edf_data, csv_data, channel_names):
    """Check if the baseline is consistent across EDF and CSV."""
    for channel in channel_names:
        edf_baseline = np.mean(edf_data[channel])
        csv_baseline = np.mean(csv_data[channel].values)
        print(f"{channel} Baseline - EDF: {edf_baseline:.2f}, CSV: {csv_baseline:.2f}")
        assert np.isclose(edf_baseline, csv_baseline, atol=0.1), f"Baseline mismatch in channel {channel}."


def convert_edf_to_csv(edf_file, csv_file):
    """Convert EDF to CSV, checking all criteria."""
    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    edf_info = {
        'sfreq': raw.info['sfreq'],
        'n_channels': raw.info['nchan'],
        'channel_names': raw.ch_names,
        'n_samples': raw.n_times
    }

    # Create a DataFrame for the EDF data
    edf_data = pd.DataFrame(raw.get_data().T, columns=edf_info['channel_names'])
    edf_data["Time (s)"] = np.arange(0, edf_info['n_samples']) / edf_info['sfreq']

    # Save to CSV format
    edf_data.to_csv(csv_file, index=False)
    print(f"Converted {edf_file} to {csv_file}")

    # Load the CSV data back for verification
    csv_data = pd.read_csv(csv_file)

    # Verify Sampling Rate and Duration
    check_sampling_rate_duration(edf_info, csv_data)

    # Verify Channel Names
    check_channel_names(edf_info['channel_names'], list(csv_data.columns[:-1]))  # Exclude 'Time (s)' column

    # Plot Signal Comparison (First 1000 samples)
    plot_signal_comparison(edf_data, csv_data, edf_info['channel_names'], edf_info['sfreq'], sample_segment=1000)

    # Check Baseline and Offset Consistency
    check_signal_baseline_offset(edf_data, csv_data, edf_info['channel_names'])

    print("All checks passed. Conversion is accurate.")


# Example usage:
edf_file = 'path_to_your_edf_file.edf'
csv_file = 'output.csv'
convert_edf_to_csv(edf_file, csv_file)
