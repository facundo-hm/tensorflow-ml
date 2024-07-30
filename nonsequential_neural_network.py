from utils import layers, Model, optimizers, losses, metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()

X, X_test, y, y_test = train_test_split(
    housing.data, housing.target, random_state=42)

X_wide, X_deep = X[:, :5], X[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]

### WIDE & DEEP NEURAL NETWORK ###
input_wide = layers.Input(shape=X_wide.shape[1:])
input_deep = layers.Input(shape=X_deep.shape[1:])
norm_layer_wide = layers.Normalization()
norm_layer_deep = layers.Normalization()
hidden_layer1 = layers.Dense(30, activation='relu')
hidden_layer2 = layers.Dense(30, activation='relu')
concat_layer = layers.Concatenate()
output_layer = layers.Dense(1)

norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = hidden_layer1(norm_deep)
hidden2 = hidden_layer2(hidden1)
concat = layers.concatenate([norm_wide, hidden2])
output = output_layer(concat)

norm_layer_wide.adapt(X_wide)
norm_layer_deep.adapt(X_deep)

model = Model(inputs=[input_wide, input_deep], outputs=[output])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=losses.mean_squared_error,
    metrics=[metrics.RootMeanSquaredError()])
model.fit((X_wide, X_deep), y, validation_split=0.2, epochs=50)

evaluate_values = model.evaluate((X_test_wide, X_test_deep), y_test)
print('evaluate_values', evaluate_values)
