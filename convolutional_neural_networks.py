from sklearn.datasets import load_sample_images
import tensorflow as tf
from utils import layers

imgs: list = load_sample_images()['images']
imgs = layers.CenterCrop(height=70, width=120)(imgs)
imgs = layers.Rescaling(scale=1/255)(imgs)
print('imgs', imgs.shape)

# Create layer with 32 filters, each of size 7x7
c_layer = layers.Conv2D(filters=32, kernel_size=(7, 7))
c_imgs = c_layer(imgs)
print('c_imgs', c_imgs.shape)

mp_layer = layers.MaxPool2D(pool_size=2)
mp_imgs = mp_layer(imgs)
print('mp_imgs', mp_imgs)

ap_layer = layers.AvgPool2D(pool_size=2)
ap_imgs = ap_layer(imgs)
print('ap_imgs', ap_imgs)

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
print('dp_imgs', dp_imgs)

gap_layer = layers.GlobalAvgPool2D()
gap_imgs = gap_layer(imgs)
print('gap_imgs', gap_imgs)
