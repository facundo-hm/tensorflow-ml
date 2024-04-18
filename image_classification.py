from typing import cast
from tensorflow import data
from tensorflow.python.keras import Sequential, losses
from tensorflow.python.keras.layers import Flatten, Dense, Softmax
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

Load_Response = tuple[data.Dataset, data.Dataset]

(X, y), (X_test, y_test) = cast(
    tuple[Load_Response, Load_Response],
    tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True))

# Scale image values to a range of 0 to 1
X: data.Dataset = X / MAX_VALUE
X_test: data.Dataset = X_test / MAX_VALUE

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])

model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(X, y, epochs=10)

model.evaluate(X_test, y_test, verbose=2)

# Attach a softmax layer to convert the model's
# linear outputs—logits—to probabilities
probability_model = Sequential([
    model, 
    Softmax()
])

# Predict the label for each image in the testing set
predictions = probability_model.predict(X_test)

# A prediction is an array of 10 numbers that represent the
# probability that the image corresponds to each class
print('Prediction: ', np.argmax(predictions[0]))
print('Label: ', y_test[0])

# Use the trained model to make a single prediction
img = X_test[1]

# tf.keras models are optimized to make predictions
# on a batch, or collection.
# Add the image to a batch where it's the only member.
img = np.expand_dims(img, 0)
img_prediction = probability_model.predict(img)

print('Prediction: ', np.argmax(img_prediction[0]))
print('Label: ', y_test[1])
