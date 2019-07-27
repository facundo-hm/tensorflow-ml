from tensorflow import keras

max_value = 255.0
max_images = 1000

# Load data
(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.fashion_mnist.load_data()
)

train_labels = train_labels[:max_images]
test_labels = test_labels[:max_images]

train_images = train_images[:max_images] / max_value
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

# Create callback
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath='models/mc_model_checkpoint.ckpt',
    save_weights_only=True,
    verbose=1
)

# Train model and save weights
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# Save entire model
model.save('models/mc_model.h5')
