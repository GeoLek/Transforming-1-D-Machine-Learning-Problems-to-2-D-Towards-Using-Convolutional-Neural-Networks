import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        print(f"GPU Available: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")

# Define paths
base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Data/Reshaping Method'
test_dir = os.path.join(base_dir, 'test')
model_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Models results/Minimal 2D CNN/Reshaping Method/training_run_1/final_model.h5'
output_file_path = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Models results/Minimal 2D CNN/Reshaping Method/training_run_1/test_metrics.txt'
confusion_matrix_path = os.path.join(os.path.dirname(output_file_path), 'confusion_matrix.png')
metrics_plot_path = os.path.join(os.path.dirname(output_file_path), 'metrics_plot.png')

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Class names
class_names = ['N', 'S', 'V', 'F', 'Q']

# Initialize ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generator for evaluation
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the model
model = load_model(model_path)

# Evaluate the model on test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true classes
true_classes = test_generator.classes

# Calculate metrics
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')
accuracy = accuracy_score(true_classes, predicted_classes)

# Print all metrics including accuracy
print(f"Accuracy: {accuracy}\nPrecision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Compute and plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, alpha=1.0, cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=300)  # Save the confusion matrix to a file
plt.close()

# Generate the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)

# Function to calculate TP, FP, FN, TN
def calculate_confusion_metrics(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, FN, TP, TN

FP, FN, TP, TN = calculate_confusion_metrics(cm)

# Format the classification report
formatted_report = f"Classification Report:\n"
formatted_report += classification_report(true_classes, predicted_classes, target_names=class_names)
formatted_report += f"\nConfusion Matrix:\n{cm}\n\n"
formatted_report += f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\n"
formatted_report += "Detailed Report:\n"

for label, metrics in report.items():
    if label == "accuracy":
        formatted_report += f"{label}:\n"
        formatted_report += f"  value: {metrics}\n"
    elif label not in ["macro avg", "weighted avg"]:
        formatted_report += f"{label}:\n"
        for metric_name, value in metrics.items():
            formatted_report += f"  {metric_name}: {value}\n"
        idx = class_names.index(label)
        formatted_report += f"  TP: {TP[idx]}\n"
        formatted_report += f"  FP: {FP[idx]}\n"
        formatted_report += f"  FN: {FN[idx]}\n"
        formatted_report += f"  TN: {TN[idx]}\n"
    else:
        formatted_report += f"{label}:\n"
        formatted_report += f"  precision: {metrics['precision']}\n"
        formatted_report += f"  recall: {metrics['recall']}\n"
        formatted_report += f"  f1-score: {metrics['f1-score']}\n"
        formatted_report += f"  support: {metrics['support']}\n"

# Save the classification report and confusion matrix to the output file
with open(output_file_path, 'w') as f:
    f.write(formatted_report)

# Print the formatted report to the console
print(formatted_report)

# Plotting and saving metrics
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Evaluation Metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig(metrics_plot_path)  # Save the metrics plot to a file
plt.close()

print("Evaluation complete. Metrics saved and confusion matrix plotted.")