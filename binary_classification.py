import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
import tensorflow_datasets as tfds

batch_size = 512

# Create subsplit for trainig
train_subsplit = tfds.Split.TRAIN.subsplit([6, 4])

# Load and split data
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_subsplit, tfds.Split.TEST),
    as_supervised=True
)

url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

# Create Keras layer
hub_layer = hub.KerasLayer(
    url,
    input_shape=[],
    dtype=tf.string,
    trainable=True
)

# Define model layers
model = keras.Sequential([
    hub_layer,
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Print model summary
model.summary()

# Configure model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train_data.shuffle(10000).batch(batch_size),
    epochs=10,
    validation_data=validation_data.batch(batch_size),
    verbose=1
)

# Evaluate model
model.evaluate(test_data.batch(batch_size), verbose=1)
