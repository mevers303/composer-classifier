# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import CountVectorizer
# fix random seed for reproducibility
numpy.random.seed(777)

from src.globals import *
from src.midi_archive import get_meta_df
from src.midi_processing import MidiFileText


X_filenames = []
X = []
y = []


all_df = get_meta_df("raw_midi")
all_df = all_df[all_df.type == 1]

composers_df = pd.DataFrame(all_df.groupby("composer").type.count())
composers_df.columns = ["works"]

valid_composers = composers_df[composers_df.works > MINIMUM_WORKS].index.values

for composer in valid_composers:

    composers_works = all_df[all_df.composer == composer]
    if composers_works.composer.count() > MAXIMUM_WORKS:
        composers_works = composers_works.sample(MAXIMUM_WORKS)

    X_filenames.extend(composers_works.index.values)
    y.extend(composers_works.composer.values)


total = len(X_filenames)
i = 0
for filename in X_filenames:
    X.append(MidiFileText(filename, all_df).to_text())
    i += 1
    progress_bar(i, total)





def tokenize(text):
    return text.split(" ")

vectorizer = CountVectorizer(tokenizer=tokenize)
X_transformed = vectorizer.fit_transform(X)


hidden_layer_size = 1000


n_words = len(vectorizer.get_feature_names())
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

