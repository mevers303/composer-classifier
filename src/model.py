# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
# fix random seed for reproducibility
np.random.seed(777)

from src.globals import *
from src.midi_archive import get_meta_df
from src.midi_processing import MidiFileText
from src.dataset import get_docs



X, y, composers, n_words = get_docs()








# create the model
model = Sequential()
model.add(InputLayer(input_shape=n_words))
model.add(LSTM(units=hidden_layer_size))
model.add(LSTM(units=hidden_layer_size))
#model.add(LSTM(units=hidden_layer_size))
model.add(Dense(units=n_composers, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

