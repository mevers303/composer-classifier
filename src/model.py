# Mark Evers
# Created: 3/30/2018
# model.py
# RNN classifier model

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense


from globals import *
from dataset import VectorGetterText, VectorGetterNHot

dataset = VectorGetterNHot("raw_midi")

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
    model.add(LSTM(units=HIDDEN_LAYER_SIZE, input_shape=(NUM_STEPS, dataset.n_features), return_sequences=True))
    model.add(LSTM(units=np.int32(HIDDEN_LAYER_SIZE / 2), return_sequences=True))
    model.add(LSTM(units=np.int32(HIDDEN_LAYER_SIZE / 4)))
    model.add(Dense(units=dataset.n_composers, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    return model



def batch_fit_model(model):

    # FIT THE MODEL
    print("Training model...")
    with open("model_log.txt", "a") as f:
        f.write("***MODEL***\n")
        f.write("Neurons: " + str(HIDDEN_LAYER_SIZE) + "\n")
        f.write("Layers: 2\n")

    for epoch in range(N_EPOCHS):
        print("EPOCH", epoch + 1, "/", N_EPOCHS)
        dataset.reset_chunks()
        progress_bar(dataset.last_train_chunk_i, dataset.n_train_files)
        while dataset.last_train_chunk_i < dataset.n_train_files:
            X, y = dataset.get_chunk(BATCH_SIZE, "train")
            loss = model.train_on_batch(X, y)
            progress_bar(dataset.last_train_chunk_i, dataset.n_train_files, text=str(loss))
            with open("model_log.txt", "a") as f:
                f.write("EPOCH {}: {}\n".format(epoch, loss))


        with open("model_log.txt", "a") as f:
            f.write("\n")
        print()  # newline

        return model



def all_fit_model(model):

    X_train, X_test, y_train, y_test = dataset.get_all_split()

    # FIT THE MODEL
    print("Training model...")
    with open("model_log.txt", "a") as f:
        f.write("***MODEL***\n")
        f.write("Neurons: " + str(HIDDEN_LAYER_SIZE) + "\n")
        f.write("Layers: 2\n")

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    with open("model_log.txt", "a") as f:
        f.write(str(history))
        f.write("\n\n")
    print()  # newline

    return model



def save_model(model, filename):

    print("Saving model to disk")
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + "model.h5")



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

        print("Epoch progress:")
        progress_bar(dataset.last_test_chunk_i, dataset.n_test_files)
        print()

    print(predictions)


    # MEASURE ACCURACY
    total = actual.shape[0]
    correct = 0
    for y, y_pred in zip(actual, predictions):
        if np.argmax(y) == np.argmax(y_pred):
            correct += 1

    return correct / total




if __name__ == "__main__":

    model = create_model()
    model = all_fit_model(model)
    save_model(model, "models/2_layer_nhot_" + str(HIDDEN_LAYER_SIZE))
    # model = load_model_from_disk()
    accuracy = get_model_accuracy(model)

    print("Accuracy:", accuracy)
