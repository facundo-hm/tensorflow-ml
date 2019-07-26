import tensorflow as tf
from tensorflow import keras
import numpy as np

word_count = 10000
pad_value = 0
sequence_maxlen = 256

# Load data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=word_count
)


def convert_to_hot_encoding(sequences, dimension):
    hot_encoded_sequences = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        hot_encoded_sequences[i, word_indices] = 1.0
    return hot_encoded_sequences


# Hot enconde data
train_data = convert_to_hot_encoding(train_data, dimension=word_count)
test_data = convert_to_hot_encoding(test_data, dimension=word_count)

# Create a validation set
validation_train_data = train_data[:10000]
partial_train_data = train_data[10000:]

validation_train_labels = train_labels[:10000]
partial_train_labels = train_labels[10000:]

# Define model
overfitted_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(word_count,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Configure model
overfitted_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

overfitted_model.summary()

# Train and monitor model
overfitted_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

# Define model layers with weight regularizer
regularized_model = keras.Sequential([
    keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu',
        input_shape=(word_count,)
    ),
    keras.layers.Dense(
        16,
        kernel_regularizer=keras.regularizers.l2(0.001),
        activation='relu'
    ),
    keras.layers.Dense(1, activation='sigmoid')
])

# Configure model
regularized_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

regularized_model.summary()

# Train and monitor model
regularized_model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_train_data, validation_train_labels),
    verbose=2
)

# Define model layers with dropout regularizer
droppedout_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(word_count,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Configure model
droppedout_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

droppedout_model.summary()

# Train and monitor model
droppedout_model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_train_data, validation_train_labels),
    verbose=2
)

# Evaluate models
regularized_model.evaluate(test_data, test_labels)
droppedout_model.evaluate(test_data, test_labels)
