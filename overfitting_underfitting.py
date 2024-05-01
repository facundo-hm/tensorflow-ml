from utils import (
    optimizers, callbacks, metrics, losses,
    Sequential, layers)
import pandas as pd
import numpy as np

SEQUENTIAL = Sequential
FEATURES = 28
N_ROWS = 100000
N_VALIDATION = 0.2
BATCH_SIZE = 500
STEPS_PER_EPOCH = (N_ROWS - (N_ROWS * N_VALIDATION)) // BATCH_SIZE
MAX_EPOCHS = 100
COLUMN_NAMES = ['class_label', 'jet_1_b-tag', 'jet_1_eta', 'jet_1_phi',
    'jet_1_pt', 'jet_2_b-tag', 'jet_2_eta', 'jet_2_phi', 'jet_2_pt',
    'jet_3_b-tag', 'jet_3_eta', 'jet_3_phi', 'jet_3_pt', 'jet_4_b-tag',
    'jet_4_eta', 'jet_4_phi', 'jet_4_pt', 'lepton_eta', 'lepton_pT',
    'lepton_phi', 'm_bb', 'm_jj', 'm_jjj', 'm_jlv', 'm_lv', 'm_wbb',
    'm_wwbb', 'missing_energy_magnitude', 'missing_energy_phi']

# gz = utils.get_file(
#     'HIGGS.csv.gz',
#     'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

data_train = pd.read_csv(
   '~/.keras/datasets/HIGGS.csv',
   names=COLUMN_NAMES,
   nrows=N_ROWS)

data_labels = data_train.pop(COLUMN_NAMES[0])

lr_schedule = optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 10,
    decay_rate=1,
    staircase=False)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train))

def compile_and_fit(
    model: SEQUENTIAL,
    optimizer=optimizers.Adam(lr_schedule),
    max_epochs=MAX_EPOCHS
):
    model.compile(
        optimizer=optimizer,
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            metrics.BinaryCrossentropy(
                from_logits=True, name='binary_crossentropy'),
            'accuracy'])

    model.summary()

    history = model.fit(
        data_train,
        data_labels,
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        validation_split=N_VALIDATION,
        callbacks=callbacks.EarlyStopping(
            monitor='val_binary_crossentropy', patience=200),
        verbose=1)

    return history

tiny_model = Sequential([
    normalizer,
    layers.Dense(16, activation='elu'),
    layers.Dense(1)])

compile_and_fit(tiny_model)
