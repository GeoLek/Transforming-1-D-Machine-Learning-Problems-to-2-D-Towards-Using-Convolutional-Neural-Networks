import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the base directory where all output files will be saved
output_base_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/output'
output_dir = os.path.join(output_base_dir, 'training_run_1')
os.makedirs(output_dir, exist_ok=True)

# Define paths for various output files
model_checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
training_history_path = os.path.join(output_dir, 'training_history.csv')

# Paths to training and validation directories
train_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/train'
validation_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Continuous Wavelet Transform (CWT)/validation'

def generate_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        label = "normal" if "normal" in file_name else "paced"
        file_paths.append(file_path)
        labels.append(label)
    return file_paths, labels

# Generate file paths and labels for training and validation
train_image_paths, train_labels = generate_file_paths_and_labels(train_dir)
validation_image_paths, validation_labels = generate_file_paths_and_labels(validation_dir)

# Convert labels to binary format
train_labels_binary = np.array([0 if label == 'normal' else 1 for label in train_labels])

# Calculate class weights for the training data
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_binary), y=train_labels_binary)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Setup generators
train_df = pd.DataFrame({'filename': train_image_paths, 'label': train_labels})
validation_df = pd.DataFrame({'filename': validation_image_paths, 'label': validation_labels})

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=None,
                                                    x_col='filename',
                                                    y_col='label',
                                                    target_size=(224, 224),
                                                    color_mode='grayscale',
                                                    batch_size=16,
                                                    class_mode='binary',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_dataframe(dataframe=validation_df,
                                                              directory=None,
                                                              x_col='filename',
                                                              y_col='label',
                                                              target_size=(224, 224),
                                                              color_mode='grayscale',
                                                              batch_size=16,
                                                              class_mode='binary',
                                                              shuffle=False)

# Building the minimal model
model = Sequential([
    Input(shape=(224, 224, 1)),  # Adjusted for grayscale input
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Model Training with class weights
history = model.fit(train_generator,
                    epochs=5,
                    validation_data=validation_generator,
                    class_weight=class_weights_dict,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
                        tf.keras.callbacks.CSVLogger(training_history_path),
                    ])

# Save the trained model
model.save(os.path.join(output_dir, 'final_model.h5'))
