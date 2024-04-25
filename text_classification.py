from typing import cast
import re
import string
from tensorflow import data, strings, constant
import tensorflow_datasets as tfds
from utils import (
    Sequential, layers, losses, optimizers,
    metrics, activations)

Load_Response = tuple[data.Dataset, data.Dataset, data.Dataset]

train_data, validation_data, test_data = cast(
    Load_Response,
    tfds.load(
        name ='imdb_reviews', 
        split=('train[:60%]', 'train[60%:]', 'test'),
        batch_size=32,
        shuffle_files=True,
        as_supervised=True))

def standardize_data(input_data):
    input_data = strings.lower(input_data)
    input_data = strings.regex_replace(input_data, '<br />', ' ')
    input_data = strings.regex_replace(
        input_data,
        '[%s]' % re.escape(string.punctuation),
        '')

    return input_data

MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
EPOCHS = 10

vectorize_layer = layers.TextVectorization(
    standardize=standardize_data,
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

# Remove lables
train_text = train_data.map(lambda x, y: x)
# Fit the state of the preprocessing layer to the dataset
vectorize_layer.adapt(train_text)

model = Sequential([
    vectorize_layer,
    layers.Embedding(MAX_FEATURES, EMBEDDING_DIM),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation=activations.sigmoid)])

model.summary()

model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer=optimizers.Adam(),
    metrics=[metrics.BinaryAccuracy(threshold=0.5)])

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS)

model.evaluate(test_data)

new_reviews = constant([
  'The movie was great!',
  'The movie was okay.',
  'The movie was terrible...'
])

prediction = model.predict(new_reviews)

print('Prediction: ', prediction)
