from typing import cast
from tensorflow import data
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense
import tensorflow_datasets as tfds
import numpy as np

MAX_VALUE = 255.0
LABEL_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

(train_images, train_labels), (test_images, test_labels) = (
    tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True)
)

train_images = cast(data.Dataset, train_images)
train_labels = cast(data.Dataset, train_labels)
test_images = cast(data.Dataset, test_images)
test_labels = cast(data.Dataset, test_labels)

# Scale image values to a range of 0 to 1
train_images = train_images / MAX_VALUE
test_images = test_images / MAX_VALUE

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

test_predictions = model.predict(test_images)

print(test_predictions[0])
print(np.argmax(test_predictions[0]))

image_prediction = np.expand_dims(test_images[0], axis=0)
single_prediction = model.predict(image_prediction)

print(single_prediction)
print(np.argmax(single_prediction))
