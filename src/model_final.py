# Mark Evers
# Created: 3/30/2018
# _model_scratchpad.py
# RNN classifier _model

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import pickle
import os

from globals import *
from file_handlers.dataset import VectorGetterNHot
from midi_handlers.midi_file import MidiFileNHot
from file_handlers.midi_archive import MidiArchive

# fix random seed for reproducibility
# np.random.seed(777)





def create_model(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=666, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.555))
    _model.add(LSTM(units=444, return_sequences=True))
    _model.add(Dropout(.333))
    _model.add(LSTM(222))
    _model.add(Dropout(.111))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(_model.summary())

    return _model



def load_from_disk(filename):

    # load json and create _model
    with open(filename + '.json', 'r') as f:
        json_str = f.read()
    _model = model_from_json(json_str)
    # load weights into new _model
    _model.load_weights(filename + ".h5")
    print("Loaded model from disk")

    return _model



def save_to_disk(_model, filename):

    print("Saving model to disk")
    # serialize _model to JSON
    _model_json = _model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(_model_json)
    # serialize weights to HDF5
    _model.save_weights(filename + ".h5")



def fit_model(_dataset, _model):

    logfile = "models/final.txt"
    X_train, X_test, y_train, y_test = _dataset.get_all_split()

    # FIT THE _model
    print("Training model...")
    with open(logfile, "a") as f:
        f.write("***Model***\n")
        f.write("Neurons: 666 -> 444 -> 222\n")
        f.write("Dropout: .555 -> .333 -> .111\n")


    history = _model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    with open(logfile, "a") as f:
        f.write(str(history))
        f.write("\n\n")
    print("")  # newline


    return _model



def kfold_eval(_dataset):

    X, y = _dataset.get_all()
    kfold = KFold(n_splits=10, shuffle=True)

    results = cross_val_score(KerasClassifier(build_fn=create_model, epochs=N_EPOCHS, batch_size=BATCH_SIZE), X, y, cv=kfold)
    print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    print(results)

    return results



def predict_one_file(_model, filename, _dataset=None):

    if _dataset:
        note_dist = _dataset.meta_df.loc[filename][MUSIC_NOTES].values
    else:
        note_dist = MidiArchive.parse_midi_meta(filename)[14:]

    mid = MidiFileNHot(filename, note_dist)
    X = np.array(mid.to_X(), dtype=np.byte)

    y_pred = _model.predict(X)
    sum_probs = y_pred.sum(axis=0)
    normed_probs = sum_probs / sum_probs.sum()

    result = np.argmax(normed_probs)

    return result, normed_probs



def eval_file_accuracy(_dataset, _model):

    y = _dataset.y_test_filenames
    y_pred = [predict_one_file(_model, filename, _dataset)[0] for filename in _dataset.X_test_filenames]
    y_pred_labels = np.array([_dataset.composers[row] for row in y_pred])

    accuracy = (y == y_pred_labels).sum() / len(y)
    precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred_labels)

    print("\nModel Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("F-Score:  ", fscore)
    print("Support:  ", support)

    return accuracy, precision, recall, fscore



class FileAccuracyCallback(Callback):

    def __init__(self, _dataset):
        super().__init__()
        self.dataset = _dataset
        self.history = []
        self.best_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):

        accuracy, precision, recall, fscore = eval_file_accuracy(self.dataset, self.model)
        self.history.append((accuracy, precision, recall, fscore))

        if accuracy > self.best_accuracy:
            print("Saving new best accuracy (", accuracy, ")")
            self.model.save_weights("models/final_{0:02d}-{1:.2f}.h5".format(epoch, accuracy))
            self.best_accuracy = accuracy


if __name__ == "__main__":

    if os.path.exists("midi/classical/dataset.pkl"):
        with open("midi/classical/dataset.pkl", "rb") as f:
            dataset = pickle.load(f)

    else:
        dataset = VectorGetterNHot("midi/classical")
        with open("midi/classical/dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)


    X_train, X_test, y_train, y_test = dataset.get_all_split()
    file_accuracy = FileAccuracyCallback(dataset)

    model = create_model(dataset)
    with open("models/final.json", "w") as f:
        f.write(model.to_json())

    model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, callbacks=[file_accuracy])


    i = 1
    for accuracy, precision, recall, fscore in file_accuracy.history:
        print("\nModel Metrics (epoch {}):".format(i))
        print("Accuracy: ", accuracy)
        print("Precision:", precision)
        print("Recall:   ", recall)
        print("F-Score:  ", fscore)
        i += 1
