from tensorflow import keras

MAX_VALUE = 255.0
MAX_IMAGES = 1000

(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.fashion_mnist.load_data()
)

test_labels = test_labels[:MAX_IMAGES]
test_images = test_images[:MAX_IMAGES] / MAX_VALUE

# Define model layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Set compilation settings
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load model weights
model.load_weights('models/mc_model_checkpoint.ckpt')
model.evaluate(test_images, test_labels)

# Load entire model
model_two = keras.models.load_model('models/mc_model.h5')
model_two.evaluate(test_images, test_labels)
