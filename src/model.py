# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy
from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(777)



hidden_layer_size = 1000


n_words = 1000
n_composers = 1000






# create the model
embedding_vecor_length = 32
model = Sequential()
# model.add(Embedding(n_words, embedding_vecor_length, input_length=n_words))
model.add(LSTM(units=hidden_layer_size))
model.add(LSTM(units=hidden_layer_size))
model.add(LSTM(units=hidden_layer_size))
model.add(Dense(units=n_composers, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

