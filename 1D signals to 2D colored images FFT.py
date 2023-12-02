import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram

def load_ecg_data(filename):
    """
    Load your ECG data from a file.
    This function needs to be adapted based on your data format.
    """
    # For example, if your data is stored in a CSV file:
    # data = np.loadtxt(filename)
    # return data
    # Placeholder for actual data loading code
    pass

def generate_spectrogram(ecg_data, fs):
    """
    Generate a spectrogram from the ECG data.
    :param ecg_data: 1D numpy array of ECG data
    :param fs: Sampling frequency
    :return: Spectrogram
    """
    f, t, Sxx = spectrogram(ecg_data, fs)
    return f, t, Sxx

def plot_colored_spectrogram(f, t, Sxx):
    """
    Plot the spectrogram with a color map.
    :param f: Frequencies
    :param t: Time
    :param Sxx: Spectrogram
    """
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.show()

# Example Usage
filename = 'path_to_your_ecg_data_file'
ecg_data = load_ecg_data(filename)
sampling_frequency = 1000  # Replace with the actual sampling frequency of your ECG data

f, t, Sxx = generate_spectrogram(ecg_data, sampling_frequency)
plot_colored_spectrogram(f, t, Sxx)
