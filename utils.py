from typing import cast
import tensorflow as tf
import keras

# Fix autocomplete issue
tf_keras = cast(keras, tf.keras)

Sequential = tf_keras.Sequential
layers = tf_keras.layers
losses = tf_keras.losses
metrics = tf_keras.metrics
activations = tf_keras.activations
optimizers = tf_keras.optimizers
utils = tf_keras.utils
callbacks = tf_keras.callbacks
regularizers = tf_keras.regularizers
Model = tf_keras.Model
constraints = tf_keras.constraints
