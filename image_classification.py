from typing import cast
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils import (
    Sequential, layers, losses, optimizers, metrics)

MAX_VALUE = 255.0
LABEL_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
    'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

Load_Response = tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]

train_data, validation_data, test_data = cast(
  Load_Response,
  tfds.load(
        'fashion_mnist',
        split=('train[:80%]', 'train[80%:]', 'test'),
        batch_size=128,
        as_supervised=True))

def normalize_img(image, label):
    # x and y must have the same dtype.
    # Normalize images from uint8 to float32
    return tf.cast(image, tf.float32) / MAX_VALUE, label

train_data = train_data.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
validation_data = validation_data.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

model = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)])

model.summary()

model.compile(
    optimizer=optimizers.Adam(0.001),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.SparseCategoricalAccuracy()])

model.fit(train_data, validation_data=validation_data, epochs=10)

model.evaluate(test_data, verbose=2)

# Attach a softmax layer to convert the model's
# linear outputs—logits—to probabilities
probability_model = Sequential([
    model, 
    layers.Softmax()
])

# Remove labels
X_test = test_data.map(lambda x, y: x)

# Predict the label for each image in the testing set
predictions = probability_model.predict(X_test)

X_batch_images, y_batch_lables = tuple(test_data.take(1))[0]

# A prediction is an array of 10 numbers that represent the
# probability that the image corresponds to each class
print('Prediction: ', np.argmax(predictions[0]))
print('Label: ', y_batch_lables[0])

# Use the trained model to make a single prediction
img = X_batch_images[1]

# tf.keras models are optimized to make predictions on a batch,
# or collection. Add the image to a batch where it's the only member.
img = np.expand_dims(img, 0)
img_prediction = probability_model.predict(img)

print('Prediction: ', np.argmax(img_prediction[0]))
print('Label: ', y_batch_lables[1])
