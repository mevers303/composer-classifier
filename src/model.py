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
from sklearn.model_selection import train_test_split


from src.globals import *
from src.midi_archive import get_meta_df
from src.midi_processing import MidiFileText
from src.dataset import get_docs

# fix random seed for reproducibility
np.random.seed(777)


doclen = 250

X, y, composers, n_words, max_doc_len = get_docs()
X_padded = np.array([sequence.pad_sequences(np.array(x[:doclen,:].T.todense()), maxlen=doclen).T for x in X])
X_train, X_test, y_train, y_test = train_test_split(X_padded, y)
n_composers = len(composers)




hidden_layer_size = 512

# create the model
model = Sequential()
model.add(LSTM(units=hidden_layer_size, input_shape=(doclen, n_words)))
# model.add(LSTM(units=hidden_layer_size, input_shape=(250, n_words)))
model.add(Dense(units=n_composers, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
