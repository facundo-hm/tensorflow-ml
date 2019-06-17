import tensorflow as tf
from tensorflow import keras
import numpy as np

word_count = 10000
pad_value = 0
sequence_maxlen = 256

# Load data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=word_count)

# Standardize train_data length
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=pad_value,
    padding='post',
    maxlen=sequence_maxlen
)

# Standardize test_data length
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=pad_value,
    padding='post',
    maxlen=sequence_maxlen
)

# Define model layers
model = keras.Sequential([
    keras.layers.Embedding(word_count, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.summary()

# Configure model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)

# Create a validation set
validation_train_data = train_data[:10000]
partial_train_data = train_data[10000:]

validation_train_labels = train_labels[:10000]
partial_train_labels = train_labels[10000:]

# Train and monitor model
model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_train_data, validation_train_labels),
    verbose=1
)

# Evaluate model
model.evaluate(test_data, test_labels)
