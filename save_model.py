from typing import cast
from tensorflow import data
from tensorflow.python.keras import Sequential, callbacks
from tensorflow.python.keras.layers import Dense, Flatten
import tensorflow_datasets as tfds

MAX_VALUE = 255.0

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

cp_callback = callbacks.ModelCheckpoint(
    filepath='models/mc_model_checkpoint.ckpt',
    save_weights_only=True,
    verbose=1
)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

model.save('models/mc_model.h5')
