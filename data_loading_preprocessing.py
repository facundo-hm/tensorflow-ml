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

def parse_csv_dataset(dataset: tf.data.Dataset):
    data: list[tf.Tensor]= []
    lables: list[tf.Tensor]= []

    for line in dataset.as_numpy_iterator():
        defs = (
            [0.] * 2
            + [tf.constant('', dtype=tf.string)]
            + [0.] * 40)
        fields = tf.io.decode_csv(line, record_defaults=defs)
        data.append(fields[1:2] + fields[3:25] + fields[26:])
        lables.append(fields[25])

    return tf.data.Dataset.from_tensor_slices((data, lables))

def create_dataset(
    dataset: tf.data.Dataset, n_readers=3, seed=SEED,
    batch_size=25, shuffle_buffer_size=20
):
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=n_readers, num_parallel_calls=n_readers)
    dataset = parse_csv_dataset(dataset).cache()
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    return dataset.repeat(50).batch(batch_size).prefetch(1)

train_dataset = create_dataset(train_csv_dataset)
valid_dataset = create_dataset(valid_csv_dataset)
test_dataset = create_dataset(test_csv_dataset)

X_train = train_dataset.map(lambda x, y: x)
y_train = train_dataset.map(lambda x, y: y)

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
model.fit(train_dataset, validation_data=valid_dataset, epochs=160)
model.evaluate(test_dataset, verbose=2)

prediction = model.predict(X_train)

