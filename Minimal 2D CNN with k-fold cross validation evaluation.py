import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define paths
test_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/k-fold cross validation/Continuous Wavelet Transform (CWT)/test' # Update this path to your test dataset
output_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/k-fold cross validation/Continuous Wavelet Transform (CWT)/output/'  # Path where models are saved
model_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if 'best_model_fold' in f or 'final_model' in f]
output_file_path = os.path.join(output_dir, 'test_metrics.txt')
# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Load Data
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
test_labels = ['normal' if 'normal' in f else 'paced' for f in test_files]

# Initialize ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Prepare the data generator for evaluation
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_files, 'label': test_labels}),
    directory=None,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=16,
    class_mode=None,  # No labels are provided to the generator
    shuffle=False  # Important for maintaining order
)

# Calculate the correct number of steps
steps_per_epoch = np.ceil(len(test_files) / 16)

# Convert test_labels to binary format before the loop
test_labels_binary = np.array([0 if label == 'normal' else 1 for label in test_labels])


# Evaluate Model
# Loop through each model and evaluate
for model_path in model_paths:
    model = load_model(model_path)
    print(f"Evaluating {os.path.basename(model_path)}")

    # Predict using the correct number of steps
    predictions = model.predict(test_generator, steps=steps_per_epoch)

    # Flatten predictions to match the length of test_labels_binary
    predicted_classes = (predictions > 0.5).astype(int).flatten()[
                        :len(test_labels_binary)]  # Ensure predicted classes match the length of test_labels_binary

    # Compute Evaluation Metrics
    accuracy = accuracy_score(test_labels_binary, predicted_classes)
    precision = precision_score(test_labels_binary, predicted_classes)
    recall = recall_score(test_labels_binary, predicted_classes)
    f1 = f1_score(test_labels_binary, predicted_classes)

    # Print metrics
    print(
        f"Model: {os.path.basename(model_path)}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n")

    # Confusion Matrix
    cm = confusion_matrix(test_labels_binary, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['normal', 'paced'], yticklabels=['normal', 'paced'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{os.path.basename(model_path)}.png"))
    plt.close()

    # Save metrics to file
    with open(output_file_path, 'a') as f:
        f.write(
            f"Model: {os.path.basename(model_path)}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\n")
