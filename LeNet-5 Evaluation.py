import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Define paths
test_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/test'
model_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/output/training_run_1/final_model.h5'
output_file_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/output/training_run_1/test_metrics.txt'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Create DataFrame for test data
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
test_labels = ['normal' if 'normal' in f else 'paced' for f in test_files]
test_df = pd.DataFrame({'filename': test_files, 'label': test_labels})

# Initialize ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generator for evaluation
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col=None,  # No labels are used here since we're manually handling them
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,  # Or adjust based on your configuration
    class_mode=None,  # No class mode since we're manually handling labels
    shuffle=False
)

# Load the model
model = load_model(model_path)

# Evaluate the model on test data
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Binary classification thresholding

# Assuming test_labels are in the correct binary format ['normal', 'paced'] => [0, 1]
true_classes = np.array([0 if label == 'normal' else 1 for label in test_labels])

# Calculate metrics
precision = precision_score(true_classes, predicted_classes, average='binary')
recall = recall_score(true_classes, predicted_classes, average='binary')
f1 = f1_score(true_classes, predicted_classes, average='binary')

# Print metrics
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save the metrics
with open(output_file_path, 'w') as f:
    f.write(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

# Optionally, save a detailed classification report
class_names = ['normal', 'paced']  # Adjust if necessary
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)
with open(output_file_path, 'a') as f:
    f.write("\nDetailed Classification Report:\n")
    f.write(report)
