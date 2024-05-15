import pandas as pd
import numpy as np
from utils import (
    Sequential, layers, optimizers, activations,
    losses, metrics, constraints)

URL = (
    'http://archive.ics.uci.edu/ml/machine-learning-databases/'
    'auto-mpg/auto-mpg.data')
COLUMN_NAMES = [
    'MPG', 'Cylinders', 'Displacement', 'Horsepower',
    'Weight', 'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(
    URL,
    names=COLUMN_NAMES,
    na_values='?',
    comment='\t',
    sep=' ',
    skipinitialspace=True)

# Drop rows with unknown values
dataset = dataset.dropna()

# Convert from numeric to categorical
dataset['Origin'] = dataset['Origin'].map(
    {1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'])

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate label from features
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

def create_hidden_layer(units: int):
    return layers.Dense(
        units,
        activation=activations.relu,
        kernel_initializer='he_normal',
        kernel_constraint=constraints.max_norm(1.))

model = Sequential([
    normalizer,
    create_hidden_layer(64),
    create_hidden_layer(64),
    layers.Dense(1)])

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.mean_squared_error,
    metrics=[metrics.mean_squared_error])

model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=100)

model.evaluate(test_features, test_labels)

prediction = model.predict(test_features)

print('Prediction: ', prediction[0])
print('Label: ', test_labels.iloc[0])
