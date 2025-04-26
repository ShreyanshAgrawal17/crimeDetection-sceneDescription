import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU before TensorFlow initializes

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from build_model import create_binary_lstm_model

# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
use_gpu = False

if gpu_devices:
    try:
        # Attempt to use GPU
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        use_gpu = True
        print("⚡ Using GPU for training")
    except:
        print("⚠️ GPU detected but not usable due to CuDNN mismatch. Falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow
        use_gpu = False

print(f"⚡ Training on: {'GPU' if use_gpu else 'CPU'}")

# Load preprocessed holistic feature data
X_train = np.load("X_train_preprocessed.npy")  # Shape: (num_samples, 100, 225)
y_train = np.load("y_train_preprocessed.npy")
X_test = np.load("X_test_preprocessed.npy")
y_test = np.load("y_test_preprocessed.npy")

# Define Model
input_shape = (100, 225)  # Updated to 100 frames per video
num_classes = y_train.shape[1]
model = create_binary_lstm_model(input_shape, num_classes)

# Define Callbacks
lr_reduction = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, verbose=1, min_lr=1e-6)  
early_stopping = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1)  

# Determine batch size dynamically (adjust if dataset is small)
batch_size = min(32, X_train.shape[0] // 10)  # Ensures at least 10 batches per epoch

# Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  
    batch_size=batch_size,  
    callbacks=[lr_reduction, early_stopping],
    verbose=1
)

# Save Trained Model
model.save("crime_detection_model.h5")
print("✅ Model successfully trained and saved as 'crime_detection_model.h5' 🚀")