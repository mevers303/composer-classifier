# Mark Evers
# Created: 3/30/2018
# dataset.py
# Functions for getting the data set

import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.globals import *
from src.midi_processing import MidiFileText



class VectorGetter():

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.meta_df = None
        self.composers = None
        self.X_filenames = None
        self.y_filenames = None
        self.last_chunk_i = 0
        self.n_files = 0

        self.y_label_encoder = None
        self.y_onehot_encoder = None

        self.get_meta_df()
        self.get_composers()
        self.get_filenames()





    def get_meta_df(self, csv_file = "meta.csv"):
        """
        Gets a meta dataframe with the proper schema from <dir>.

        :param csv_file: The name of the csv file.
        :return: A pandas dataframe with the metadate.
        """
        self.meta_df = pd.read_csv(os.path.join(self.base_dir, csv_file), index_col="filename")
        return self.meta_df



    def get_composers(self):
        """
        Returns a list of composers that have at least MINIMUM_WORKS pieces.

        :return: A list of composers
        """
        composers_df = pd.DataFrame(self.meta_df.groupby("composer").type.count())
        composers_df.columns = ["works"]

        valid_composers = composers_df[composers_df.works > MINIMUM_WORKS].index.values

        print("Found", len(valid_composers), "composers:", ", ".join(valid_composers))

        self.composers = valid_composers
        self.y_label_encoder = LabelEncoder().fit(self.composers)
        self.y_onehot_encoder = OneHotEncoder().fit(self.y_label_encoder.transform(self.composers).reshape(-1, 1))

        return valid_composers



    def get_filenames(self):
        """
        Gets a list of filenames for each composer and the appropriate label.

        :return: <list of tracks as text>, <list of labels>
        """
        self.X_filenames = []
        self.y_filenames = []

        for composer in self.composers:

            composers_works = self.meta_df[self.meta_df.composer == composer]
            if composers_works.composer.count() > MAXIMUM_WORKS:
                composers_works = composers_works.sample(MAXIMUM_WORKS)

            self.X_filenames.extend(composers_works.index.values)
            self.y_filenames.extend(composers_works.composer.values)

        self.n_files = len(self.X_filenames)
        print("Found a total of", self.n_files, "MIDI files!")

        return self.X_filenames, self.y_filenames




class VectorGetterText(VectorGetter):
    
    def __init__(self, base_dir = "midi"):
        super().__init__(base_dir)

        self.vectorizer_pickle = os.path.join(self.base_dir, "text_vectorizer.pkl")
        self.vectorizer = None

        self.get_vectorizer()
        self.n_features = len(self.vectorizer.get_feature_names())


    def get_vectorizer(self):

        if os.path.exists(self.vectorizer_pickle):
            print("Loading vectorizer from", self.vectorizer_pickle, "...")
            with open(self.vectorizer_pickle, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            self.train_vectorizer()


    def get_text_chunk(self, X_chunk_filenames, y_chunk_filenames):

        X_text = []
        y_text = []


        print("Loading chunk of MIDI files...")
        total = len(X_chunk_filenames)
        complete = 0

        for filename, composer in zip(X_chunk_filenames, y_chunk_filenames):

            text = MidiFileText(filename, self.meta_df).to_text()
            X_text.extend(text)
            y_text.extend([composer] * len(text))

            self.last_chunk_i += 1
            complete += 1
            progress_bar(complete, total)

        y_text = self.y_label_encoder.transform(y_text).reshape(-1, 1)
        y_text = self.y_onehot_encoder.transform(y_text)

        return X_text, y_text


    @staticmethod
    def tokenize(text):
        """
        Simple tokenize function to be used in the CountVectorizer
        :param text: The text to be tokenized
        :return: A list of tokens
        """
        return text.split(" ")



    def get_docs_chunk(self, chunk_size):
        """
        Easy wrapper function to get all the docs and their labels

        :return: docs: list of docs, y: list of docs' labels, composers: list of composers, n_features: number of features
        """

        X_chunk_filenames = self.X_filenames[self.last_chunk_i:self.last_chunk_i + chunk_size]
        y_chunk_filenames = self.y_filenames[self.last_chunk_i:self.last_chunk_i + chunk_size]

        X_chunk_text, y = self.get_text_chunk(X_chunk_filenames, y_chunk_filenames)
        print("Transforming corpus...")
        docs = []
        # max_doc_len = 0
        for track in X_chunk_text:
            tokens = self.tokenize(track)
            docs.append(self.vectorizer.transform(tokens))
            # track_len = len(tokens)
            # if track_len > max_doc_len:
            #     max_doc_len = track_len


        print(len(docs), "individual tracks loaded!")

        return docs, y


    def train_vectorizer(self):

        print("Learning vocabulary...")
        vocab = set()

        i = 0
        total = self.meta_df.index.size
        for file in self.meta_df.index.values:
            mid = MidiFileText(file, self.meta_df)
            text = mid.to_text()

            for track in text:
                vocab.update(track.split(" "))

            i += 1
            progress_bar(i, total)
        
        vocab = list(vocab)

        print("Fitting vectorizer...")
        self.vectorizer = CountVectorizer(tokenizer=VectorGetterText.tokenize, max_features=TEXT_MAXIMUM_FEATURES).fit(vocab)

        print("Saving", self.vectorizer_pickle, "...")
        with open(self.vectorizer_pickle, "wb") as f:
            pickle.dump(self.vectorizer, f)



if __name__ == "__main__":

    getter = VectorGetterText("raw_midi")
    while getter.last_chunk_i < getter.n_files:
        X, y = getter.get_docs_chunk(CHUNK_SIZE)
