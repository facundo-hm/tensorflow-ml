import tensorflow as tf
from tensorflow import keras
import numpy as np

max_value = 255.0

# Load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define possible labels
label_names = [
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

# Scale image values to a range of 0 to 1
train_images = train_images / max_value
test_images = test_images / max_value

# Define model layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Set compilation settings
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Predict images
test_predictions = model.predict(test_images)

print(test_predictions[0])
print(np.argmax(test_predictions[0]))

# Predict one image
image_prediction = np.expand_dims(test_images[0], axis=0)

single_prediction = model.predict(image_prediction)

print(single_prediction)
print(np.argmax(single_prediction))
