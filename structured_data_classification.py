import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
from sklearn.model_selection import train_test_split

url = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(url)

train_df, test_df = train_test_split(dataframe, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)


def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    labels = dataframe.copy().pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    return ds.batch(batch_size)


batch_size = 32
train_ds = dataframe_to_dataset(train_df, batch_size=batch_size)
val_ds = dataframe_to_dataset(val_df, False, batch_size)
test_ds = dataframe_to_dataset(test_df, False, batch_size)

# Numeric columns
feature_columns = [
    feature_column.numeric_column(header)
    for header
    in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']
]

# Bucketized columns
age_buckets = feature_column.bucketized_column(
    feature_columns[0],
    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
)
feature_columns.append(age_buckets)

# Categorical columns
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal',
    ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# Embedding columns
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# Crossed feature columns
crossed_feature = feature_column.crossed_column(
    [age_buckets, thal],
    hash_bucket_size=1000
)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

model = keras.Sequential([
  keras.layers.DenseFeatures(feature_columns),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
    run_eagerly=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.evaluate(test_ds)
