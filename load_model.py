from tensorflow import keras

max_value = 255.0
max_images = 1000

# Load data
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.fashion_mnist.load_data()
)

test_labels = test_labels[:max_images]
test_images = test_images[:max_images] / max_value

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

# Evaluate model
model.evaluate(test_images, test_labels)

# Load entire model
model_two = keras.models.load_model('models/mc_model.h5')

# Evaluate model
model_two.evaluate(test_images, test_labels)
