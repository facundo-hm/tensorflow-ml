from typing import cast
import tensorflow as tf
from utils import (
   Sequential, layers, activations, optimizers, losses, metrics)
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo

Load_Response = tuple[
    tuple[tf.data.Dataset, tf.data.Dataset], DatasetInfo]

(ds_train, ds_test), info = cast(
    Load_Response,
    tfds.load(
        'penguins/processed',
        split=['train[:80%]', 'train[80%:]'],
        batch_size=32,
        as_supervised=True,
        with_info=True))

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam()

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)

    return loss_value, tape.gradient(
        loss_value, model.trainable_variables)

model = Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(10, activation=activations.relu),
    layers.Dense(10, activation=activations.relu),
    layers.Dense(3)])

NUM_EPOCHS = 100

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss_avg = metrics.Mean()
    epoch_accuracy = metrics.SparseCategoricalAccuracy()

    for x, y in ds_train:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Add current batch loss
        epoch_loss_avg.update_state(loss_value)
        # Compare predicted label to actual label
        epoch_accuracy.update_state(y, model(x, training=True))

    if epoch % 10 == 0:
        print('\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}\n'.format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
