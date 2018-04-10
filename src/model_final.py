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

    results = cross_val_score(KerasClassifier(build_fn=create_model, epochs=N_EPOCHS, batch_size=BATCH_SIZE), X, y, cv=kfold)
    print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    print(results)

    return results



def predict_one_file(_dataset, _model, filename):

    mid = MidiFileNHot(filename, _dataset.meta_df)
    X = np.array(mid.to_X(), dtype=np.byte)

    y_pred = _model.predict(X)
    sum_probs = y_pred.sum(axis=0)
    normed_probs = sum_probs / sum_probs.sum()

    result = np.zeros(shape=(y_pred.shape[1]))
    result[np.argmax(sum_probs)] = 1

    return result, normed_probs



def eval_file_accuracy(_dataset, _model):

    y = _dataset.y_test_filenames
    y_pred = [predict_one_file(_dataset, _model, filename)[0] for filename in _dataset.X_test_filenames]
    y_labels = np.array([_dataset.composers(np.argmax(row)) for row in y])
    y_pred_labels = np.array([_dataset.composers(np.argmax(row)) for row in y_pred])

    accuracy = (y_labels == y_pred_labels).sum()
    precision, recall, fscore, support = precision_recall_fscore_support(y_labels, y_pred_labels)

    print("\nModel Metrics:")
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:   ", recall)
    print("F-Score:  ", fscore)
    print("Support:  ", support)

    return accuracy, precision, recall, fscore


if __name__ == "__main__":

    dataset = VectorGetterNHot("midi/classical")
    model = create_model(dataset)
    # model = load_from_disk("models/final")
    model = fit_model(dataset, model, pickle_file="100-120_works_split.pkl")
    save_to_disk(model, "models/final")
    accuracy, precision, recall, fscore = eval_file_accuracy(dataset, model)
