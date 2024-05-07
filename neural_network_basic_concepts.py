from typing import cast
import tensorflow as tf
from utils import (
   Sequential, layers, activations, optimizers, losses, metrics)
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo
import numpy as np

SequentialType = Sequential
Dataset = tf.data.Dataset

NUM_EPOCHS = 10
BATCH_SIZE = 32
CLASS_NAMES = ['Adélie', 'Chinstrap', 'Gentoo']

Load_Response = tuple[
    tuple[Dataset, Dataset], DatasetInfo]

(ds_train, ds_test), info = cast(
    Load_Response,
    tfds.load(
        'penguins/processed',
        split=['train[:80%]', 'train[80%:]'],
        batch_size=BATCH_SIZE,
        as_supervised=True,
        with_info=True))

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam()

def loss(model: SequentialType, X: Dataset, y: Dataset, training=True):
    y_ = model(X, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model: SequentialType, X: Dataset, y: Dataset):
    with tf.GradientTape() as tape:
        loss_value = loss(model, X, y)

    return loss_value, tape.gradient(
        loss_value, model.trainable_variables)

model = Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(10, activation=activations.relu),
    layers.Dense(10, activation=activations.relu),
    layers.Dense(3)])

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss_avg = metrics.Mean()
    epoch_accuracy = metrics.SparseCategoricalAccuracy()

    for X, y in ds_train:
        # Optimize the model
        loss_value, grads = grad(model, X, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Add current batch loss
        epoch_loss_avg.update_state(loss_value)
        # Compare predicted label to actual label
        epoch_accuracy.update_state(y, model(X, training=True))

    if epoch % 10 == 0:
        print('\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}\n'.format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

test_accuracy = metrics.Accuracy()

# Evaluate the model
for X, y in ds_test:
    logits = model(X, training=False)
    print(logits)
    prediction = np.argmax(logits, axis=1)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

ds_predict = tf.convert_to_tensor([
    [0.3, 0.8, 0.4, 0.5],
    [0.4, 0.1, 0.8, 0.5],
    [0.7, 0.9, 0.8, 0.4]
])

predictions = model(ds_predict, training=False)

# Make predictions
for i, logits in enumerate(predictions):
    class_idx = np.argmax(logits)
    percentages = layers.Softmax()(logits)
    class_percentage = percentages[class_idx] * 100
    class_name = CLASS_NAMES[class_idx]
    
    print("Example {} prediction: {} ({:4.1f}%)".format(
        i, class_name, class_percentage))
