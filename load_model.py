from typing import cast
from tensorflow import data
from tensorflow.python.keras import Sequential, layers, models
import tensorflow_datasets as tfds

MAX_VALUE = 255.0
MAX_IMAGES = 1000

(train_images, train_labels), (test_images, test_labels) = (
    tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True)
)

test_images = cast(data.Dataset, test_images)
test_labels = cast(data.Dataset, test_labels)

test_labels = test_labels[:MAX_IMAGES]
test_images = test_images[:MAX_IMAGES] / MAX_VALUE

model = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights('models/mc_model_checkpoint.ckpt')
model.evaluate(test_images, test_labels)

model_two = models.load_model('models/mc_model.h5')
model_two.evaluate(test_images, test_labels)
