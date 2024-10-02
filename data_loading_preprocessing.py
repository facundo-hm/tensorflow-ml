import tensorflow as tf

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

for line in dataset.take(5):
    print(line)
