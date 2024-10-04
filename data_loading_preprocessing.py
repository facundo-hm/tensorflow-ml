import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

BASE_PATH = './datasets/car_price/'
train_files = ['train_01.csv', 'train_02.csv', 'train_03.csv']
valid_files = ['valid_01.csv', 'valid_02.csv', 'valid_03.csv']
test_files = ['test_01.csv', 'test_02.csv', 'test_03.csv']

train_dataset = tf.data.Dataset.list_files(
    BASE_PATH + 'train_*.csv', seed=42)

n_readers = 3
dataset = train_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath),
    cycle_length=n_readers)

car_names: list[tf.Tensor] = []

def parse_csv_line(line):
    defs = [0.] * 2 + [tf.constant('', dtype=tf.string)] + [0.] * 40
    fields = tf.io.decode_csv(line, record_defaults=defs)
    car_names.append(fields[2])
    fields[25], fields[42] = fields[42], fields[25]

    return fields[1:]

parsed_dataset = []

for line in dataset.as_numpy_iterator():
    parsed_line = parse_csv_line(line)
    parsed_dataset.append(parsed_line)

car_names = tf.strings.strip(car_names)
le = LabelEncoder()
car_names_encoded = le.fit_transform(car_names)

def add_encoded_name(idx, line):
    line[1] = tf.constant(car_names_encoded[idx], dtype=tf.float32)
    return tf.stack(line[:-1]), tf.stack(line[-1:])

new_dataset: list[tuple]= []

for i, line in enumerate(parsed_dataset):
    X_y = add_encoded_name(i, line)
    new_dataset.append(X_y)

print(new_dataset)
