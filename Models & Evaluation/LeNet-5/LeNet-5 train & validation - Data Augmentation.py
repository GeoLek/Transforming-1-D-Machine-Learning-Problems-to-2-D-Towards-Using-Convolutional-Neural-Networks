import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# Define the base directory where all output files will be saved
output_base_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/output'

# Create an output directory specific to this training run
output_dir = os.path.join(output_base_dir, 'training_run_1')
os.makedirs(output_dir, exist_ok=True)

# Define paths for various output files
model_checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
training_history_path = os.path.join(output_dir, 'training_history.csv')

# Paths to training and validation directories
train_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/train'
validation_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Short-Time Fourier Transform (STFT)/validation'

# Function to generate file paths and labels
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

# Data generators with augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_df = pd.DataFrame({'filename': train_image_paths, 'label': train_labels})
validation_df = pd.DataFrame({'filename': validation_image_paths, 'label': validation_labels})

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=4,
    class_mode='binary',
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=4,
    class_mode='binary',
    shuffle=False,
)

# LeNet-5 Model definition adjusted for 224x224 images
input_tensor = Input(shape=(224, 224, 1))
x = Conv2D(6, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
x = AveragePooling2D()(x)
x = Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
x = AveragePooling2D()(x)
x = Flatten()(x)
x = Dense(120, activation='relu')(x)
x = Dense(84, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Model Training with class weights
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    class_weight=class_weights_dict,  # Apply the class weights here
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(training_history_path),
    ]
)

# Save the trained model
model.save(os.path.join(output_dir, 'final_model.h5'))
