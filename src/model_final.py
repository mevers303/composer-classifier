# Mark Evers
# Created: 3/30/2018
# _model_scratchpad.py
# RNN classifier _model

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold

from globals import *
from file_handlers.dataset import VectorGetterNHot
from midi_handlers.midi_file import MidiFileNHot
from file_handlers.midi_archive import MidiArchive

# fix random seed for reproducibility
# np.random.seed(777)





def create_model0(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=666, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.555))
    _model.add(LSTM(units=444))
    _model.add(Dropout(.333))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    print(_model.summary())

    return _model


def create_model1(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=666, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.333))
    _model.add(LSTM(units=444))
    _model.add(Dropout(.222))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    print(_model.summary())

    return _model


def create_model2(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=444, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.555))
    _model.add(LSTM(units=333))
    _model.add(Dropout(.333))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    print(_model.summary())

    return _model



def create_model3(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=1024, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.555))
    _model.add(LSTM(units=512))
    _model.add(Dropout(.333))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    print(_model.summary())

    return _model



def create_model4(_dataset):

    # CREATE THE _model
    _model = Sequential()
    _model.add(LSTM(units=444, input_shape=(NUM_STEPS, _dataset.n_features), return_sequences=True))
    _model.add(Dropout(.555))
    _model.add(LSTM(units=333, return_sequences=True))
    _model.add(Dropout(.333))
    _model.add(LSTM(222))
    _model.add(Dropout(.111))
    _model.add(Dense(units=_dataset.n_composers, activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
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



def fit_model(_dataset, _model, pickle_file="nhot_split.pkl"):

    logfile = "models/final.txt"
    X_train, X_test, y_train, y_test = _dataset.get_all_split(pickle_file)

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

    results = cross_val_score(KerasClassifier(build_fn=create_model0, epochs=N_EPOCHS, batch_size=BATCH_SIZE), X, y, cv=kfold)
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


if __name__ == "__main__":

    # model = load_from_disk("models/final")
    dataset = VectorGetterNHot("midi/classical")

    # model = create_model0(dataset)
    # model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    # save_to_disk(model, "models/final_0")
    # accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)

    model = create_model1(dataset)
    model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    save_to_disk(model, "models/final_1")
    accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)

    model = create_model2(dataset)
    model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    save_to_disk(model, "models/final_2")
    accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)

    model = create_model3(dataset)
    model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    save_to_disk(model, "models/final_3")
    accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)

    model = create_model4(dataset)
    model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    save_to_disk(model, "models/final_4")

    accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)
    # predict_one_file(model, "midi/classical/Bach/bach_846.mid")
