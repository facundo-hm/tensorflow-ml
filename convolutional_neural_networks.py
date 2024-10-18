from typing import cast
from sklearn.datasets import load_sample_images
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from utils import layers, Sequential

imgs: list = load_sample_images()['images']
imgs = layers.CenterCrop(height=70, width=120)(imgs)
imgs = layers.Rescaling(scale=1/255)(imgs)
# print('imgs', imgs.shape)

# Create layer with 32 filters, each of size 7x7
c_layer = layers.Conv2D(filters=32, kernel_size=(7, 7))
c_imgs = c_layer(imgs)
# print('c_imgs', c_imgs.shape)

mp_layer = layers.MaxPool2D(pool_size=2)
mp_imgs = mp_layer(imgs)
# print('mp_imgs', mp_imgs)

ap_layer = layers.AvgPool2D(pool_size=2)
ap_imgs = ap_layer(imgs)
# print('ap_imgs', ap_imgs)

class DepthPool(layers.Layer):
    def __init__(self, pool_size=3, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        # Number of channel groups
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat(
            [shape[:-1], [groups, self.pool_size]], axis=0)
        
        # Gather the channels by groups and pick the highest value
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)

dp_layer = DepthPool(pool_size=3)
dp_imgs = dp_layer(imgs)
# print('dp_imgs', dp_imgs)

gap_layer = layers.GlobalAvgPool2D()
gap_imgs = gap_layer(imgs)
# print('gap_imgs', gap_imgs)

def create_conv_2d_layer(**kwargs):
    default_args = {
        'kernel_size': 3,
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal'
    } | kwargs

    return layers.Conv2D(**default_args)

Load_Response = tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]

train_data, validation_data, test_data = cast(
    Load_Response,
    tfds.load(
        'fashion_mnist',
        split=('train[:80%]', 'train[80%:]', 'test'),
        batch_size=128, as_supervised=True))

rescaling = layers.Rescaling(scale=.01/255)

train_data = train_data.map(lambda x, y: (rescaling(x), y))
validation_data = validation_data.map(lambda x, y: (rescaling(x), y))

# Basic CNN
model = Sequential([
    create_conv_2d_layer(
        filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    layers.MaxPool2D(),
    create_conv_2d_layer(filters=128),
    create_conv_2d_layer(filters=128),
    layers.MaxPool2D(),
    create_conv_2d_layer(filters=256),
    create_conv_2d_layer(filters=256),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(
        units=128, activation='relu', kernel_initializer='he_normal'),
    layers.Dropout(0.5),
    layers.Dense(
        units=64, activation='relu', kernel_initializer='he_normal'),
    layers.Dropout(0.5),
    layers.Dense(units=10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='nadam', metrics=['accuracy'])
model.fit(
    train_data, validation_data=validation_data, epochs=10)
