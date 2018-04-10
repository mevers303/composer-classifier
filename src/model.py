# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


from globals import *
from dataset import VectorGetterText, VectorGetterNHot, VectorGetterNHotTimeSeries

dataset = VectorGetterNHot("midi/classical")
logfile = "models/log.txt"

# fix random seed for reproducibility
np.random.seed(777)




def load_model_from_disk():

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model



def create_model():

    # CREATE THE MODEL
    model = Sequential()
    model.add(LSTM(units=666, input_shape=(NUM_STEPS, dataset.n_features), return_sequences=True))
    model.add(Dropout(.555))
    model.add(LSTM(units=444, return_sequences=True))
    model.add(Dropout(.333))
    model.add(LSTM(222))
    model.add(Dropout(.111))
    model.add(Dense(units=dataset.n_composers, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', 'accuracy'])
    print(model.summary())

    return model



def batch_fit_model(model):

    # FIT THE MODEL
    print("Training model...")
    with open(logfile, "a") as f:
        f.write("***MODEL***\n")
        f.write("Neurons: " + str(HIDDEN_LAYER_SIZE) + "\n")
        f.write("Layers: 2\n")

    for epoch in range(N_EPOCHS):

        print("EPOCH", epoch + 1, "/", N_EPOCHS)

        dataset.reset_chunks()
        progress_bar(dataset.last_train_chunk_i, dataset.n_train_files)
        while dataset.last_train_chunk_i < dataset.n_train_files:

            X, y = dataset.get_chunk(BATCH_FILES, "train")
            this_batch_len = X.shape[0]
            shuffled_i = np.arange(this_batch_len)
            np.random.shuffle(shuffled_i)
            X = X[shuffled_i]
            y = y[shuffled_i]

            this_batch_i = 0
            while this_batch_i < this_batch_len:
                loss = model.train_on_batch(X[this_batch_i:this_batch_i + BATCH_SIZE], y[this_batch_i:this_batch_i + BATCH_SIZE])
                this_batch_i += BATCH_SIZE
                progress_bar(this_batch_i, this_batch_len, text=str(loss))



        with open(logfile, "a") as f:
            f.write("EPOCH {}: {}\n".format(epoch, get_model_accuracy(model)))


        save_model(model, "text_model", epoch)

        print()  # newline

    return model



def all_fit_model(model):

    X_train, X_test, y_train, y_test = dataset.get_all_split()

    # FIT THE MODEL
    print("Training model...")
    with open(logfile, "a") as f:
        f.write("***MODEL***\n")
        f.write("Neurons: " + str(HIDDEN_LAYER_SIZE) + "\n")
        f.write("Layers: 2\n")


    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    with open(logfile, "a") as f:
        f.write(str(history))
        f.write("\n\n")
    print()  # newline

    return model



def kfold_eval():

    X, y = dataset.get_all()
    kfold = KFold(n_splits=10, shuffle=True, random_state=777)

    results = cross_val_score(KerasClassifier(build_fn=create_model, epochs=N_EPOCHS, batch_size=BATCH_SIZE), X, y, cv=kfold)
    print("Result: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    print(results)




def save_model(model, filename, epoch=0):

    print("Saving model to disk")
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + "_e{}.h5".format(epoch))



def get_model_accuracy(model):

    # MAKE PREDICTIONS
    predictions = None
    actual = None
    first_round = True

    print("Evaluating model...")
    progress_bar(dataset.last_test_chunk_i, dataset.n_test_files)

    while dataset.last_test_chunk_i < dataset.n_test_files:
        X, y = dataset.get_chunk(BATCH_SIZE, "test")
        if first_round:
            actual = y
            predictions = model.predict(X)
        else:
            actual = np.append(actual, y, axis=0)
            predictions = np.append(predictions, model.predict(X), axis=0)


    # MEASURE ACCURACY
    total = actual.shape[0]
    correct = 0
    for y, y_pred in zip(actual, predictions):
        if np.argmax(y) == np.argmax(y_pred):
            correct += 1

    return correct / total




if __name__ == "__main__":

    model = create_model()

    # model = load_model_from_disk()
    if type(dataset) == VectorGetterNHot:
        model = all_fit_model(model)
        save_model(model, "models/final")
    elif type(dataset) == VectorGetterText:
        model = batch_fit_model(model)

    accuracy = get_model_accuracy(model)

    print("Accuracy:", accuracy)

    # kfold_eval()
