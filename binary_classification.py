import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

BATCH_SIZE = 512
URL = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'

train_subsplit = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(
    name='imdb_reviews',
    split=(train_subsplit, tfds.Split.TEST),
    as_supervised=True
)

hub_layer = hub.KerasLayer(
    URL,
    input_shape=[],
    dtype=tf.string,
    trainable=True
)
model = keras.Sequential([
    hub_layer,
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data.shuffle(10000).batch(BATCH_SIZE),
    epochs=10,
    validation_data=validation_data.batch(BATCH_SIZE),
    verbose=1
)

model.evaluate(test_data.batch(BATCH_SIZE), verbose=1)
