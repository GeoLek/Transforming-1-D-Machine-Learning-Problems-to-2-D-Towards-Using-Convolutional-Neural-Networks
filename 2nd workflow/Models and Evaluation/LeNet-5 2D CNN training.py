import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        print(f"GPU Available: {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Using CPU.")

# Define the base directory where all output files will be saved
output_base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Models results/LeNet-5 2D CNN/Fast Fourier Transform (FFT)'
output_dir = os.path.join(output_base_dir, 'training_run_1')
os.makedirs(output_dir, exist_ok=True)

# Define paths for various output files
model_checkpoint_path = os.path.join(output_dir, 'model_checkpoint.h5')
training_history_path = os.path.join(output_dir, 'training_history.csv')
metrics_plot_path = os.path.join(output_dir, 'metrics_plot.png')

# Paths to training and validation directories
base_dir = '/home/orion/Geo/Projects/Transforming-1-D-Machine-Learning-Problems-to-2-D-Towards-Using-Convolutional-Neural-Networks/Data/Fast Fourier Transform (FFT)'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Setup generators
train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(224, 224),
                                                    color_mode='grayscale',
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                              target_size=(224, 224),
                                                              color_mode='grayscale',
                                                              batch_size=16,
                                                              class_mode='categorical',
                                                              shuffle=False)

# Building the model using LeNet-5 architecture
model = Sequential([
    Input(shape=(224, 224, 1)),
    Conv2D(6, (5, 5), activation='tanh'),
    AveragePooling2D((2, 2)),
    Conv2D(16, (5, 5), activation='tanh'),
    AveragePooling2D((2, 2)),
    Conv2D(120, (5, 5), activation='tanh'),
    Flatten(),
    Dense(84, activation='tanh'),
    Dense(5, activation='softmax')  # Adjust the number of units as per the number of classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Model Training
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=validation_generator,
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
                        tf.keras.callbacks.CSVLogger(training_history_path),
                        early_stopping_callback
                    ])

# Save the trained model
model.save(os.path.join(output_dir, 'final_model.h5'))

# Plotting and saving metrics
history_df = pd.DataFrame(history.history)
history_df.to_csv(training_history_path, index=False)

plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history_df['accuracy'], label='Train Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_df['loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.savefig(metrics_plot_path)
plt.show()

print("Training complete. Model saved and metrics plotted.")
