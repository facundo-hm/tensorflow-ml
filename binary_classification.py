from typing import cast
from tensorflow import string, data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow_hub as hub
import tensorflow_datasets as tfds

BATCH_SIZE = 512
EMBEDDING = 'https://tfhub.dev/google/nnlm-en-dim50/2'

train_data, validation_data, test_data = tfds.load(
    name ='imdb_reviews', 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_data = cast(data.Dataset, train_data)
validation_data = cast(data.Dataset, validation_data)
test_data = cast(data.Dataset, test_data)

hub_layer = hub.KerasLayer(
    EMBEDDING,
    input_shape=[],
    dtype=string,
    trainable=True)
model = Sequential([
    hub_layer,
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(
    train_data.shuffle(10000).batch(BATCH_SIZE),
    epochs=10,
    validation_data=validation_data.batch(BATCH_SIZE),
    verbose=1)

model.evaluate(test_data.batch(BATCH_SIZE), verbose=1)
