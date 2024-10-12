import tensorflow as tf
from utils import (
    Sequential, losses, optimizers, layers,
    activations, metrics, Model)
import numpy as np

BASE_PATH = './datasets/car_price/'
SEED = 42

train_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'train_*.csv', seed=SEED)
test_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'test_*.csv', seed=SEED)
valid_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'valid_*.csv', seed=SEED)

def decode_line(line: str):
    defs = (
        [0.] * 2 + [tf.constant('', dtype=tf.string)] + [0.] * 40)
    return tf.io.decode_csv(line, record_defaults=defs)

def create_dataset(dataset: tf.data.Dataset, n_readers=3):
    return dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=n_readers, num_parallel_calls=n_readers)

def parse_csv_dataset(
    dataset: tf.data.Dataset, n_readers=3, seed=SEED,
    batch_size=25, shuffle_buffer_size=20
):
    dataset = create_dataset(dataset, n_readers)

    data: list[tf.Tensor]= []
    lables: list[tf.Tensor]= []

    for line in dataset.as_numpy_iterator():
        fields = decode_line(line)
        # Remove Car ID, Car Model and Price columns
        data.append(fields[1:2] + fields[3:25] + fields[26:])
        # Price column
        lables.append(fields[25])

    dataset = tf.data.Dataset.from_tensor_slices(
        (data, lables)).cache()
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    return dataset.repeat(50).batch(batch_size).prefetch(1)

train_dataset = parse_csv_dataset(train_csv_dataset)
valid_dataset = parse_csv_dataset(valid_csv_dataset)
test_dataset = parse_csv_dataset(test_csv_dataset)

X_train = train_dataset.map(lambda x, y: x)

normalizer = layers.Normalization()
normalizer.adapt(X_train)

train_dataset = train_dataset.map(lambda X, y: (normalizer(X), y))
valid_dataset = valid_dataset.map(lambda X, y: (normalizer(X), y))
test_dataset = test_dataset.map(lambda X, y: (normalizer(X), y))

model = Sequential([
    layers.Dense(41, activation=activations.relu,
        kernel_initializer='he_normal'),
    layers.Dense(41, activation=activations.relu,
        kernel_initializer='he_normal'),
    layers.Dense(1)
])

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()])
# model.fit(train_dataset, validation_data=valid_dataset, epochs=160)
# model.evaluate(test_dataset, verbose=2)

def extract_categories():
    ds = train_csv_dataset.concatenate(
        test_csv_dataset).concatenate(valid_csv_dataset)
    ds = create_dataset(ds)

    # categories: list[tf.Tensor]= [
    #     decode_line(line)[2] for line in ds.as_numpy_iterator()]
    
    data: list[tf.Tensor]= []
    categories: list[tf.Tensor]= []
    lables: list[tf.Tensor]= []

    for line in ds.as_numpy_iterator():
        fields = decode_line(line)
        # Remove Car ID, Car Model and Price columns
        data.append(fields[1:2] + fields[3:25] + fields[26:])
        # Car Model column
        categories.append(fields[2])
        # Price column
        lables.append(fields[25])

    return data, tf.strings.strip(categories), lables

X_num, X_cat, y = extract_categories()
X_train_num, X_train_cat, y_train = X_num[:150], X_cat[:150], y[:150]
X_valid_num, X_valid_cat, y_valid = X_num[150:], X_cat[150:], y[150:]

print('X_train_num', len(X_train_num[0]))

test_car_names = np.array([
    'volkswagen dasher', 'toyota corolla tercel',
    'subaru trezia', 'vw dasher', 'toyota corona liftback',
    'subaru tribeca', 'vw rabbit', 'toyota corolla',
    'subaru dl', 'volkswagen rabbit', 'toyota starlet',
    'subaru dl', 'volkswagen rabbit custom', 'toyota tercel',
    'toyota corona mark ii', 'volkswagen dasher',
    'toyota corolla', 'toyota corona', 'volvo 145e (sw)',
    'toyota cressida', 'toyota corolla 1200', 'volvo 144ea',
    'toyota corolla', 'volvo 244dl', 'toyota celica gt',
    'mazda glc', 'volvo 245', 'mazda rx-7 gs', 'volvo 264gl',
    'peugeot 504 (sw)', 'buick electra 225 custom',
    'volvo diesel', 'peugeot 504', 'buick century luxus (sw)'])

sl = layers.StringLookup()
sl.adapt(X_cat)

embed_model = Sequential([
    sl,
    layers.Embedding(input_dim=sl.vocabulary_size(), output_dim=2)
])

num_input = layers.Input(shape=[40], name='num')
cat_input = layers.Input(shape=[], dtype=tf.string, name='cat')

cat_embeddings = embed_model(cat_input)
encoded_inputs = layers.concatenate([num_input, cat_embeddings])
outputs = layers.Dense(1)(encoded_inputs)

model = Model(inputs=[num_input, cat_input], outputs=[outputs])
model.compile(loss='mse', optimizer='sgd')

model.fit(
    (X_train_num, X_train_cat), y_train, epochs=5,
    validation_data=((X_valid_num, X_valid_cat), y_valid))

