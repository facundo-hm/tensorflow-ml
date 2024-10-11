import tensorflow as tf
from utils import (
    Sequential, losses, optimizers, layers, activations, metrics)

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

    categories: list[tf.Tensor]= [
        decode_line(line)[2] for line in ds.as_numpy_iterator()]

    return tf.strings.strip(categories)

car_names = extract_categories()
print('car_names', car_names)
