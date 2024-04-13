from typing import cast
from tensorflow import data
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Dense, Dropout
import tensorflow_datasets as tfds
import numpy as np

WORD_COUNT = 10000
PAD_VALUE = 0
SEQUENCE_MAXLEN = 256

(train_data, train_labels), (test_data, test_labels) = (
    tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        as_supervised=True)
)

train_data = cast(data.Dataset, train_data)
train_labels = cast(data.Dataset, train_labels)
test_data = cast(data.Dataset, test_data)
test_labels = cast(data.Dataset, test_labels)

def convert_to_hot_encoding(sequences, dimension):
    hot_encoded_sequences = np.zeros((len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        hot_encoded_sequences[i, word_indices] = 1.0

    return hot_encoded_sequences

train_data = convert_to_hot_encoding(train_data, dimension=WORD_COUNT)
test_data = convert_to_hot_encoding(test_data, dimension=WORD_COUNT)

validation_train_data = train_data[:10000]
partial_train_data = train_data[10000:]
validation_train_labels = train_labels[:10000]
partial_train_labels = train_labels[10000:]

overfitted_model = Sequential([
    Dense(512, activation='relu', input_shape=(WORD_COUNT,)),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

overfitted_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

overfitted_model.summary()

overfitted_model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(test_data, test_labels),
    verbose=2
)

regularized_model = Sequential([
    Dense(
        16,
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu',
        input_shape=(WORD_COUNT,)
    ),
    Dense(
        16,
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu'
    ),
    Dense(1, activation='sigmoid')
])

regularized_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

regularized_model.summary()

regularized_model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_train_data, validation_train_labels),
    verbose=2
)

droppedout_model = Sequential([
    Dense(16, activation='relu', input_shape=(WORD_COUNT,)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

droppedout_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'binary_crossentropy']
)

droppedout_model.summary()

droppedout_model.fit(
    partial_train_data,
    partial_train_labels,
    epochs=20,
    batch_size=512,
    validation_data=(validation_train_data, validation_train_labels),
    verbose=2
)

regularized_model.evaluate(test_data, test_labels)
droppedout_model.evaluate(test_data, test_labels)
