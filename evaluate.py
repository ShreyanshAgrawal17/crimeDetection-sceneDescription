import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU
print("⚠️ GPU is not available or mismatched. Running on CPU.")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, Bidirectional,
    Lambda, Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# #  Define your original model function again here
def create_binary_lstm_model(input_shape=(100, 225), num_classes=2):
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)

    attention_data = Dense(64, activation='tanh')(x)
    attention_scores = Dense(1)(attention_data)
    attention_scores = Lambda(lambda t: tf.nn.softmax(t, axis=1))(attention_scores)

    x = Multiply()([x, attention_scores])
    x = Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    x = LayerNormalization()(x)

    x = Dense(128, activation="swish")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="swish")(x)

    outputs = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 🛠️ Rebuild the model and load weights
model = create_binary_lstm_model()
model.load_weights("crime_detection_model.h5")
print("✅ Model weights loaded successfully.")

# 📥 Load test data
X_test = np.load("X_test_preprocessed.npy")
y_test = np.load("y_test_preprocessed.npy")

# X_train = np.load("X_train_preprocessed.npy")
# y_train = np.load("y_train_preprocessed.npy")

# 🏷️ Class labels
class_labels = ["Shoplifting", "Vandalism"]

# 📊 Evaluate the model
# train_loss, train_acc = model.evaluate(X_train, y_train, verbose=1)
# print(f"✅ Train Accuracy: {train_acc * 100:.2f}%")

# 📊 Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# 🔍 Predictions and performance
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n🔍 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 📉 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("✅ Confusion Matrix plotted successfully.")