import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Define paths
data_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/k-fold cross validation/Recurrence Plots/train'  # Update this path
output_dir = '/home/orion/Geo/Projects/Transforming-1D-CNNs-to-2D-CNNs/k-fold cross validation/Recurrence Plots/output'  # Update this path

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to load images and labels
def load_dataset(directory, target_size=(224, 224)):
    images = []
    labels = []
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size, color_mode='grayscale')
        image = img_to_array(image)
        label = 'normal' if 'normal' in os.path.basename(image_path) else 'paced'
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Learning rate schedule function
def lr_schedule(epoch, lr):
    decay_rate = 0.001
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Load dataset
images, labels = load_dataset(data_dir)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Define the K-Fold Cross-Validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(images, labels_encoded):
    model = Sequential([
        Input(shape=(224, 224, 1)),  # Adjusted for grayscale input
        Conv2D(8, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),  # Adjust dropout rate as needed
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint_path = os.path.join(output_dir, f'best_model_fold_{fold_no}.h5')
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

    # Fit data to model
    history = model.fit(images[train], labels_categorical[train],
                        batch_size=8,
                        epochs=5,  # Adjust epochs as needed
                        verbose=1,
                        validation_data=(images[test], labels_categorical[test]),
                        callbacks=[model_checkpoint, lr_scheduler, early_stopping])

    # Generate generalization metrics
    scores = model.evaluate(images[test], labels_categorical[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    fold_no += 1

# Optionally, save the final model
final_model_path = os.path.join(output_dir, 'final_model.h5')
model.save(final_model_path)
