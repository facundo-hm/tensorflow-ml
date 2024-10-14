from sklearn.datasets import load_sample_images
from utils import layers

imgs = load_sample_images()['images']
imgs = layers.CenterCrop(height=70, width=120)(imgs)
imgs = layers.Rescaling(scale=1/255)(imgs)
print('imgs', imgs.shape)

# Create layer with 32 filters, each of size 7x7
conv_layer = layers.Conv2D(filters=32, kernel_size=(7, 7))
conv_imgs = conv_layer(imgs)
print('conv_imgs', conv_imgs.shape)
