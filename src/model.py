# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence


from globals import *
from dataset import VectorGetterText

dataset = VectorGetterText("raw_midi")

# fix random seed for reproducibility
np.random.seed(777)







hidden_layer_size = 1024

# create the model
model = Sequential()
model.add(LSTM(units=hidden_layer_size, input_shape=(NUM_STEPS, dataset.n_features)))
# model.add(LSTM(units=hidden_layer_size, input_shape=(NUM_STEPS, dataset.n_features), return_sequences=True))
# model.add(LSTM(units=hidden_layer_size, return_sequences=True))
# model.add(LSTM(units=hidden_layer_size))
model.add(Dense(units=dataset.n_composers, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())




# FIT THE MODEL
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
print("Training model...")
progress_bar(dataset.last_test_chunk_i, dataset.n_train_files)
while dataset.last_train_chunk_i < dataset.n_train_files:
    X, y = dataset.get_docs_chunk(CHUNK_SIZE, "train")
    model.train_on_batch(X, y)
    progress_bar(dataset.last_train_chunk_i, dataset.n_train_files)


print("Saving model to disk")
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")




# MAKE PREDICTIONS
predictions = None
actual = None
first_round = True
while dataset.last_test_chunk_i < dataset.n_test_files:
    X, y = dataset.get_docs_chunk(CHUNK_SIZE, "test")
    if first_round:
        actual = np.append(actual, y, axis=0)
        predictions = np.append(predictions, model.predict(X), axis=0)
        first_round = False
    else:
        actual = y
        predictions = model.predict(X)

print(predictions)


# MEASURE ACCURACY
total = actual.shape[0]
correct = 0
for y, y_pred in zip(actual, predictions):
    if np.array_equal(y, y_pred):
        correct += 1

print("Accuracy:", correct / total)
