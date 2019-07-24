import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Download file
dataset_file = keras.utils.get_file(
    "auto-mpg.data",
    (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/auto-mpg/auto-mpg.data"
    )
)

# Define column names
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']

# Load dataset
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

# Apply one-hot encoding
dataset = dataset.assign(
    USA=lambda row: (row['Origin'] == 1) * 1.0,
    Europe=lambda row: (row['Origin'] == 2) * 1.0,
    Japan=lambda row: (row['Origin'] == 3) * 1.0
).drop(['Origin'], axis=1)

# Split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate label from features
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Get data statistics
train_stats = train_dataset.describe().transpose()

# Normalize data
normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']


# Define model layers
model = keras.Sequential([
    keras.layers.Dense(
        64,
        activation='relu',
        input_shape=[len(train_dataset.keys())]
    ),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Configure model
model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    metrics=['mae', 'mse']
)

model.summary()

model.fit(
  normed_train_data,
  train_labels,
  epochs=1000,
  validation_split=0.2,
  verbose=1,
  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
)

model.evaluate(
    normed_test_data,
    test_labels,
    verbose=1
)
