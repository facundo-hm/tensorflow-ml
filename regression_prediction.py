from typing import Callable
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras import Sequential, callbacks, optimizers
from tensorflow.python.keras.layers import Dense
import pandas as pd

dataset_file = data_utils.get_file(
    "auto-mpg.data",
    (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/auto-mpg/auto-mpg.data"
    )
)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(
    dataset_file,
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True
)

# Drop rows with unknown values
dataset = dataset.dropna()

def create_encoding_func(value: int) -> Callable[[pd.Series], float]:
    return lambda row: (row['Origin'] == value) * 1.0

# Apply one-hot encoding
dataset = dataset.assign(
    USA=create_encoding_func(1),
    Europe=create_encoding_func(2),
    Japan=create_encoding_func(3)
).drop(['Origin'], axis=1)

# Split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate label from features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_stats = train_dataset.describe().transpose()

def normalize_data(dataset: pd.DataFrame):
    return (dataset - train_stats['mean']) / train_stats['std']

normed_train_data = normalize_data(train_dataset)
normed_test_data = normalize_data(test_dataset)

model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=[len(train_dataset.keys())]
    ),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(
    loss='mse',
    optimizer=optimizers.RMSprop(0.001),
    metrics=['mae', 'mse']
)

model.summary()

model.fit(
  normed_train_data,
  train_labels,
  epochs=1000,
  validation_split=0.2,
  verbose=1,
  callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10)]
)

model.evaluate(
    normed_test_data,
    test_labels,
    verbose=1
)
