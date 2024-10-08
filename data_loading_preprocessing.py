import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import (
    Sequential, losses, optimizers, layers, activations, metrics)
import numpy as np

BASE_PATH = './datasets/car_price/'
SEED = 42

train_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'train_*.csv', seed=SEED)
test_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'test_*.csv', seed=SEED)
valid_csv_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'valid_*.csv', seed=SEED)

def parse_csv_dataset(dataset: tf.data.Dataset):
    new_dataset = []
    car_names: list[tf.Tensor] = []

    for line in dataset.as_numpy_iterator():
        defs = (
            [0.] * 2
            + [tf.constant('', dtype=tf.string)]
            + [0.] * 40)
        fields = tf.io.decode_csv(line, record_defaults=defs)
        car_names.append(fields[2])
        fields[25], fields[42] = fields[42], fields[25]
        new_dataset.append(fields[1:])

    return new_dataset, car_names

def add_encoded_name(
    dataset: list[tf.Tensor], car_names: list[tf.Tensor]
):
    data: list[tf.Tensor]= []
    lables: list[tf.Tensor]= []
    car_names = tf.strings.strip(car_names)
    le = LabelEncoder()
    car_names_encoded = le.fit_transform(car_names)

    for i, line in enumerate(dataset):
        line[1] = tf.constant(car_names_encoded[i], dtype=tf.float32)
        data.append(line[:-1])
        lables.append(line[-1:][0])
    
    return tf.data.Dataset.from_tensor_slices((data, lables))

def create_dataset(
    dataset: tf.data.Dataset, n_readers=3, seed=SEED,
    batch_size=25, shuffle_buffer_size=20
):
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=n_readers, num_parallel_calls=n_readers)
    dataset, car_names = parse_csv_dataset(dataset)
    dataset = add_encoded_name(dataset, car_names).cache()
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    return dataset.repeat(50).batch(batch_size).prefetch(1)

train_dataset = create_dataset(train_csv_dataset)
valid_dataset = create_dataset(valid_csv_dataset)
test_dataset = create_dataset(test_csv_dataset)

X_train = train_dataset.map(lambda x, y: x)
y_train = train_dataset.map(lambda x, y: y)

normalizer = layers.Normalization()
normalizer.adapt(X_train)

model = Sequential([
    normalizer,
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
model.fit(train_dataset, validation_data=valid_dataset, epochs=160)
model.evaluate(test_dataset)

prediction = model.predict(X_train)

