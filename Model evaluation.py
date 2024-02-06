import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Define paths
test_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/test'
model_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/output/training_run_1/final_model.h5'
output_file_path = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/output/training_run_1/test_metrics.txt'
confusion_matrix_path = os.path.join(os.path.dirname(output_file_path), 'confusion_matrix.png')  # Path to save confusion matrix

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Here, define the class names used in your classification task
class_names = ['normal', 'paced']  # Adjust if necessary

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
    batch_size=16,  # Adjusted to match training configuration
    class_mode=None,  # No class mode since we're manually handling labels
    shuffle=False
)

# Load the model
model = load_model(model_path)

# Evaluate the model on test data
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Binary classification thresholding

# Convert test_labels to binary format
true_classes = np.array([0 if label == 'normal' else 1 for label in test_labels])

# Calculate metrics
precision = precision_score(true_classes, predicted_classes, average='binary')
recall = recall_score(true_classes, predicted_classes, average='binary')
f1 = f1_score(true_classes, predicted_classes, average='binary')
accuracy = accuracy_score(true_classes, predicted_classes)

# Print all metrics including accuracy
print(f"Accuracy: {accuracy}\nPrecision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Compute and plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(confusion_matrix_path)  # Save the confusion matrix to a file
plt.close()

# Save the metrics, including accuracy, to the file
with open(output_file_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

# Optionally, append a detailed classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)
with open(output_file_path, 'a') as f:
    f.write("\nDetailed Classification Report:\n")
    f.write(report)