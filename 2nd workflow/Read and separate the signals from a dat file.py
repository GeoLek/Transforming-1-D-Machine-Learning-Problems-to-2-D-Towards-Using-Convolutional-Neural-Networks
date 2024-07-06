import wfdb
import numpy as np

# Load the record (this reads the .dat, .hea, and .atr files)
record = wfdb.rdrecord('103', pn_dir='path_to_your_data')

# Extract the signals
signals = record.p_signal

# Separate the signals
signal_ml2 = signals[:, 0]  # MLII signal (first channel)
signal_v2 = signals[:, 1]   # V2 signal (second channel)

# Display some information about the signals
print(f"MLII Signal: {signal_ml2[:10]}")  # Print first 10 samples of MLII signal
print(f"V2 Signal: {signal_v2[:10]}")    # Print first 10 samples of V2 signal

# Optional: Save the separated signals to CSV for further use
np.savetxt('103_ml2.csv', signal_ml2, delimiter=',')
np.savetxt('103_v2.csv', signal_v2, delimiter=',')
