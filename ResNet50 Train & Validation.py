import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define the base directory where all output files will be saved
output_base_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method/output'

# Create an output directory specific to this training run
output_dir = os.path.join(output_base_dir, 'training_run_1')
os.makedirs(output_dir, exist_ok=True)

# Define paths for various output files
model_checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
training_history_path = os.path.join(output_dir, 'training_history.csv')
log_path = os.path.join(output_dir, 'training_log.txt')

# Paths to training and validation directories
train_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method/train'
validation_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/Dimension Transformation/Reshaping Method/validation'

# Function to generate file paths and labels
def generate_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.startswith("normal"):
            file_paths.append(file_path)
            labels.append("normal")
        elif file_name.startswith("paced"):
            file_paths.append(file_path)
            labels.append("paced")
    return file_paths, labels

# Generate file paths and labels for training and validation
train_image_paths, train_labels = generate_file_paths_and_labels(train_dir)
validation_image_paths, validation_labels = generate_file_paths_and_labels(validation_dir)

# Creating DataFrames that include paths and labels
train_df = pd.DataFrame({
    'filename': train_image_paths,
    'label': train_labels
})

validation_df = pd.DataFrame({
    'filename': validation_image_paths,
    'label': validation_labels
})

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Adjust the data generators to use the DataFrame with labels
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,  # Paths are absolute
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',  # Correct for binary classification
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=None,  # Paths are absolute
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',  # Correct for binary classification
    shuffle=False,
)

# Modify ResNet50 Model for grayscale input
input_tensor = Input(shape=(224, 224, 1))  # Adjust for grayscale images
base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification output layer

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(training_history_path),
    ]
)

# Save the final trained model
model.save(os.path.join(output_dir, 'final_model.h5'))