from typing import cast
import tensorflow as tf
from tensorflow import string, data
import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tf_keras = cast(keras, tf.keras)
(Sequential, layers, losses, metrics, activations, optimizers) = (
    tf_keras.Sequential, tf_keras.layers, tf_keras.losses,
    tf_keras.metrics, tf_keras.activations, tf_keras.optimizers)

BATCH_SIZE = 512
EMBEDDING = 'https://tfhub.dev/google/nnlm-en-dim50/2'

Load_Response = tuple[data.Dataset, data.Dataset, data.Dataset]

train_data, validation_data, test_data = cast(
    Load_Response,
    tfds.load(
        name ='imdb_reviews', 
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True))

hub_layer = hub.KerasLayer(
    EMBEDDING,
    input_shape=[],
    dtype=string,
    trainable=True)

model = Sequential()
model.add(hub_layer)
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data,
    verbose=1)

model.evaluate(test_data, verbose=1)
