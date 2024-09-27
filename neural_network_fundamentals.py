from typing import cast
import tensorflow as tf
from utils import (
   Sequential, layers, optimizers, losses, metrics,
   Model, activations)
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
        'penguins/processed', split=['train[:80%]', 'train[80%:]'],
        batch_size=BATCH_SIZE, as_supervised=True, with_info=True))

class DenseLayer(layers.Layer):
    def __init__(self, units: int, activation: str=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # Add one weight per neuron
        self.kernel = self.add_weight(
            name='karnel', shape=(int(input_shape[-1]), self.units),
            initializer='glorot_normal')
        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros')

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 'units': self.units,
            'activation': activations.serialize(self.activation)}

class CustomModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_1 = DenseLayer(4, 'relu')
        self.hidden_2 = DenseLayer(10, 'relu')
        self.out = DenseLayer(3)
        # Keep track of the reconstruction error during training
        self.reconstruction_mean = metrics.Mean(
            name='reconstruction_error')
        
    def build(self, input_shape):
        n_inputs = input_shape[-1]
        # Reconstruct the inputs of the model
        self.reconstruct = layers.Dense(n_inputs)

    def call(self, X, training=False):
        Z = self.hidden_1(X)
        Z = self.hidden_2(Z)
        # Produce the reconstruction
        reconstruction = self.reconstruct(Z)
        # Compute the reconstruction loss.
        # Preserve as much information as possible through
        # the hidden layers.
        recon_loss = tf.reduce_mean(
            tf.square(reconstruction - X))
        # Add reconstruction loss to the model's list of losses.
        # The hyperparameter ensures that it doesn't
        # dominate the main loss.
        self.add_loss(0.05 * recon_loss)

        if training:
            # Update reconstruction metric
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)

        return self.out(Z)

model = CustomModel()
model.summary()

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True)

def grad(model: SequentialType, X: Dataset, y: Dataset, training=True):
    with tf.GradientTape() as tape:
        y_ = model(X, training=training)
        loss_value = loss_object(y_true=y, y_pred=y_)

    # Compute gradients of loss_values with respect
    # to trainable_variables 
    loss_grad = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, loss_grad, y_

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_loss_avg = metrics.Mean()
    epoch_accuracy = metrics.SparseCategoricalAccuracy()

    for X, y in ds_train:
        # Optimize the model
        loss_value, grads, y_ = grad(model, X, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Add current batch loss
        epoch_loss_avg.update_state(loss_value)
        # Compare predicted label to actual label
        epoch_accuracy.update_state(y, y_)

    if epoch % 10 == 0:
        print('\nEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}\n'.format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

test_accuracy = metrics.Accuracy()

# Evaluate the model
for X, y in ds_test:
    logits = model(X, training=False)
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
