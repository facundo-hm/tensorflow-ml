import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

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
        data.append(tf.stack(line[:-1]))
        lables.append(tf.stack(line[-1:]))
    
    return tf.data.Dataset.from_tensor_slices((data, lables))

def create_dataset(
    dataset: tf.data.Dataset, n_readers=3, seed=SEED,
    batch_size=32, shuffle_buffer_size=20
):
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=n_readers)
    dataset, car_names = parse_csv_dataset(dataset)
    dataset = add_encoded_name(dataset, car_names)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)

    return dataset.batch(batch_size).prefetch(1)

train_dataset = create_dataset(train_csv_dataset)
print(train_dataset)
