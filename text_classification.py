from typing import cast
import re
import string
from tensorflow import data, strings
from keras.layers import TextVectorization
import tensorflow_datasets as tfds

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

vectorize_layer = TextVectorization(
    standardize=standardize_data,
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

# Remove lables
train_text = train_data.map(lambda x, y: x)
# Fit the state of the preprocessing layer to the dataset
vectorize_layer.adapt(train_text)
