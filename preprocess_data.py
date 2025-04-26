import numpy as np
from tensorflow.keras.utils import to_categorical

# Load extracted features
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy")

# Consistent Frame Count
MAX_FRAMES = 100  # Same as in feature extraction
FEATURES_PER_FRAME = 225  

# Standardizing Shape of Input Data
def preprocess_data(X, max_frames):
    return np.array([
        np.pad(x, ((0, max_frames - x.shape[0]), (0, 0)), mode='constant') if x.shape[0] < max_frames else x[:max_frames]
        for x in X
    ])

X_train = preprocess_data(X_train, MAX_FRAMES)
X_test = preprocess_data(X_test, MAX_FRAMES)

# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Save preprocessed data
np.save("X_train_preprocessed.npy", X_train)
np.save("y_train_preprocessed.npy", y_train)
np.save("X_test_preprocessed.npy", X_test)
np.save("y_test_preprocessed.npy", y_test)

print(f"✅ Preprocessed X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"✅ Preprocessed X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
