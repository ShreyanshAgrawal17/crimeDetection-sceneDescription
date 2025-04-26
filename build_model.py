import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, Bidirectional,
    Lambda, Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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
