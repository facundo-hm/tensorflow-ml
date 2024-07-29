from utils import layers, Model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()

X, X_test, y, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42)

### WIDE & DEEP NEURAL NETWORK ###
normalization_layer = layers.Normalization()
hidden_layer1 = layers.Dense(30, activation='relu')
hidden_layer2 = layers.Dense(30, activation='relu')
concat_layer = layers.Concatenate()
output_layer = layers.Dense(1)

input_ = layers.Input(shape=X_train.shape[1:])
normalized = normalization_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
# Concatenate the input and the second hidden layer’s output
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)

model = Model(inputs=[input_], outputs=[output])
