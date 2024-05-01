import tensorflow as tf
from tensorflow import data
from utils import (
    utils, optimizers, callbacks, metrics, losses,
    Sequential, layers)
import pandas as pd
import numpy as np

Load_Response = tuple[data.Dataset, data.Dataset, data.Dataset]

FEATURES = 28
N_ROWS = 100000
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = 160

# gz = utils.get_file(
#     'HIGGS.csv.gz',
#     'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

data_train = pd.read_csv(
   '~/.keras/datasets/HIGGS.csv',
   names=['class_label', 'jet_1_b-tag', 'jet_1_eta', 'jet_1_phi', 'jet_1_pt', 'jet_2_b-tag', 'jet_2_eta', 'jet_2_phi', 'jet_2_pt', 'jet_3_b-tag', 'jet_3_eta', 'jet_3_phi', 'jet_3_pt', 'jet_4_b-tag', 'jet_4_eta', 'jet_4_phi', 'jet_4_pt', 'lepton_eta', 'lepton_pT', 'lepton_phi', 'm_bb', 'm_jj', 'm_jjj', 'm_jlv', 'm_lv', 'm_wbb', 'm_wwbb', 'missing_energy_magnitude', 'missing_energy_phi'],
   nrows=N_ROWS
)

data_labels = data_train.pop('class_label')

lr_schedule = optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH * 10,
  decay_rate=1,
  staircase=False)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(data_train))

def compile_and_fit(model, optimizer=None, max_epochs=100):
    if optimizer is None:
        optimizer = optimizers.Adam(lr_schedule)

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
        validation_split=0.2,
        callbacks=callbacks.EarlyStopping(
            monitor='val_binary_crossentropy', patience=200),
        verbose=1
        )

    return history

tiny_model = Sequential([
    normalizer,
    layers.Dense(16, activation='elu'),
    layers.Dense(1)])

compile_and_fit(tiny_model)
