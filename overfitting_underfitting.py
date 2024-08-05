from utils import (
    optimizers, callbacks, metrics, losses, Sequential,
    layers, regularizers, utils, constraints)
import pandas as pd
import numpy as np

FEATURES = 28
N_ROWS = 100000
N_VALIDATION = 0.2
BATCH_SIZE = 500
STEPS_PER_EPOCH = (N_ROWS - (N_ROWS * N_VALIDATION)) // BATCH_SIZE
MAX_EPOCHS = 20
COLUMN_NAMES = ['class_label', 'jet_1_b-tag', 'jet_1_eta', 'jet_1_phi',
    'jet_1_pt', 'jet_2_b-tag', 'jet_2_eta', 'jet_2_phi', 'jet_2_pt',
    'jet_3_b-tag', 'jet_3_eta', 'jet_3_phi', 'jet_3_pt', 'jet_4_b-tag',
    'jet_4_eta', 'jet_4_phi', 'jet_4_pt', 'lepton_eta', 'lepton_pT',
    'lepton_phi', 'm_bb', 'm_jj', 'm_jjj', 'm_jlv', 'm_lv', 'm_wbb',
    'm_wwbb', 'missing_energy_magnitude', 'missing_energy_phi']

utils.get_file(
    'HIGGS.csv.gz',
    'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

X = pd.read_csv(
    '~/.keras/datasets/HIGGS.csv',
    names=COLUMN_NAMES,
    nrows=N_ROWS)

y = X.pop(COLUMN_NAMES[0])

# Gradually reduce the learning rate during training
lr_schedule = optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 10,
    decay_rate=1,
    staircase=False)

'''
Weight regularization puts constraints on the complexity of
a network by forcing its weights only to take small values,
which makes the distribution of weight values more "regular".

Dropout consists of randomly "dropping out" (i.e. set to zero)
a number of output features of the layer during training.
'''
model = Sequential([
    layers.Input((FEATURES,)),
    layers.Dense(
        512, activation='elu',
        kernel_regularizer=regularizers.l2(0.0001),
        kernel_constraint=constraints.max_norm(1.)),
    layers.Dropout(0.5),
    layers.Dense(
        512, activation='elu',
        kernel_regularizer=regularizers.l2(0.0001),
        kernel_constraint=constraints.max_norm(1.)),
    layers.Dropout(0.5),
    layers.Dense(
        512, activation='elu',
        kernel_regularizer=regularizers.l2(0.0001),
        kernel_constraint=constraints.max_norm(1.)),
    layers.Dropout(0.5),
    layers.Dense(
        512, activation='elu',
        kernel_regularizer=regularizers.l2(0.0001),
        kernel_constraint=constraints.max_norm(1.)),
    layers.Dropout(0.5),
    layers.Dense(1)
])

model.compile(
    optimizer=optimizers.Adam(lr_schedule),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        metrics.BinaryCrossentropy(
            from_logits=True, name='binary_crossentropy'),
        'accuracy'])

model.summary()

model.fit(
    X, y, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
    validation_split=N_VALIDATION,
    callbacks=callbacks.EarlyStopping(
        monitor='val_binary_crossentropy', patience=200),
    verbose=1)

# Implement Monte Carlo Dropout technique that boosts dropout models
# and provides better uncertainty estimates.
# Make 10 predictions over the test set
y_mcd_probas = np.stack([model(X, training=True) for _ in range(10)])
# Average over the first dimension
y_mcd = y_mcd_probas.mean(axis=0)

print('Predict: ', model.predict(X[:1]), y[:1])
print('Predict MCD: ', y_mcd[:1])
