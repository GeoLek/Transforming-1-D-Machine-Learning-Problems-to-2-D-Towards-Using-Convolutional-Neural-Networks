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

# Read the annotations
annotations = wfdb.rdann('103', 'atr', pn_dir='path_to_your_data')

# Extract annotation sample indices and types
ann_sample_indices = annotations.sample  # The indices of the annotations in the signal
ann_symbols = annotations.symbol  # The annotation symbols (e.g., types of beats)

# Display some information about the annotations
print(f"Annotation samples: {ann_sample_indices[:10]}")
print(f"Annotation symbols: {ann_symbols[:10]}")

# Map annotation symbols to human-readable descriptions (if needed)
# You can use the WFDB annotation types reference for more details
annotation_map = {
    'N': 'Normal beat',
    'V': 'Premature ventricular contraction',
    # Add other mappings as needed
}

# Display annotations with their descriptions
for i in range(len(ann_sample_indices[:10])):
    print(f"Sample index: {ann_sample_indices[i]}, Symbol: {ann_symbols[i]}, Description: {annotation_map.get(ann_symbols[i], 'Unknown')}")

# Optional: Save the separated signals to CSV for further use
np.savetxt('103_ml2.csv', signal_ml2, delimiter=',')
np.savetxt('103_v2.csv', signal_v2, delimiter=',')
