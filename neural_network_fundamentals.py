from typing import cast
import tensorflow as tf
from utils import (
   Sequential, layers, activations, optimizers, losses, metrics, Model)
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo
import numpy as np

SequentialType = Sequential
Dataset = tf.data.Dataset

NUM_EPOCHS = 20
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
optimizer = optimizers.SGD(
    learning_rate=0.001, momentum=0.9, nesterov=True)

def loss(model: SequentialType, X: Dataset, y: Dataset, training=True):
    y_ = model(X, training=training)
    return loss_object(y_true=y, y_pred=y_)

def grad(model: SequentialType, X: Dataset, y: Dataset):
    with tf.GradientTape() as tape:
        loss_value = loss(model, X, y)

    # Compute gradients of loss_values with respect
    # to trainable_variables 
    loss_grad = tape.gradient(loss_value, model.trainable_variables)

    return loss_value, loss_grad

class DenseLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.matmul

    def build(self, input_shape):
        # Add one weight per neuron
        self.kernel = self.add_weight(
            name='karnel',
            shape=(int(input_shape[-1]), self.units),
            initializer='glorot_normal')
        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros')

    def call(self, inputs):
        return self.activation(inputs, self.kernel) + self.bias

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__(name='')
        self.denselayer_1 = DenseLayer(4)
        self.denselayer_2 = DenseLayer(10)
        self.denselayer_3 = DenseLayer(3)

    def call(self, input_tensor):
        x = self.denselayer_1(input_tensor)
        x = activations.relu(x)
        x = self.denselayer_2(x)
        x = activations.relu(x)

        return self.denselayer_3(x)

model = CustomModel()
model.summary()

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
