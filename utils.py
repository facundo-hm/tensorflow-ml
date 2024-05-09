from typing import cast
import tensorflow as tf
import keras

# Fix autocomplete issue
tf_keras = cast(keras, tf.keras)
(Sequential, layers, losses, metrics,
activations, optimizers, utils, callbacks,
regularizers, Model) = (
    tf_keras.Sequential,
    tf_keras.layers,
    tf_keras.losses,
    tf_keras.metrics,
    tf_keras.activations,
    tf_keras.optimizers,
    tf_keras.utils,
    tf_keras.callbacks,
    tf_keras.regularizers,
    tf_keras.Model)
