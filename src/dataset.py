# Mark Evers
# Created: 3/30/2018
# dataset.py
# Functions for getting the data set

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.globals import *
from src.midi_archive import get_meta_df
from src.midi_processing import MidiFileText



def get_composers(df):
    """
    Returns a list of composers that have at least MINIMUM_WORKS pieces.

    :param df: meta dataframe
    :return: A list of composers
    """
    composers_df = pd.DataFrame(df.groupby("composer").type.count())
    composers_df.columns = ["works"]

    valid_composers = composers_df[composers_df.works > MINIMUM_WORKS].index.values

    print("Found", len(valid_composers), "composers:", ", ".join(valid_composers))
    return valid_composers


def get_text(composers, df):
    """
    Gets a list of tracks for each composer as a text string.

    :param composers: List of composers
    :param df: meta dataframe
    :return: <list of tracks as text>, <list of labels>
    """
    X_filenames = []
    y_filenames = []

    for composer in composers:

        composers_works = df[df.composer == composer]
        if composers_works.composer.count() > MAXIMUM_WORKS:
            composers_works = composers_works.sample(MAXIMUM_WORKS)

        X_filenames.extend(composers_works.index.values)
        y_filenames.extend(composers_works.composer.values)


    X_text = []
    y_text = []

    print("Loading MIDI files...")
    total = len(X_filenames)
    i = 0
    for filename, composer in zip(X_filenames, y_filenames):

        text = MidiFileText(filename, df).to_text()
        X_text.extend(text)
        y_text.extend([composer] * len(text))

        i += 1
        progress_bar(i, total)

    label_encoder = LabelEncoder().fit(composers)
    y_text = label_encoder.transform(y_text).reshape(-1, 1)
    y_text = OneHotEncoder().fit_transform(y_text)

    return X_text, y_text



def tokenize(text):
    """
    Simple tokenize function to be used in the CountVectorizer
    :param text: The text to be tokenized
    :return: A list of tokens
    """
    return text.split(" ")



def get_docs():
    """
    Easy wrapper function to get all the docs and their labels

    :return: docs: list of docs, y: list of docs' labels, composers: list of composers, n_features: number of features
    """

    df = get_meta_df("raw_midi")
    print("All files:   ", df.type.count())
    df = df[df.type == 1]
    print("Type 1 files:", df.type.count())

    composers = get_composers(df)
    X_text, y = get_text(composers, df)

    print("Training vectorizer...")
    vectorizer = CountVectorizer(tokenizer=tokenize, max_features=1000).fit(X_text)

    print("Transforming corpus...")
    docs = []
    max_doc_len = 0
    for track in X_text:
        tokens = tokenize(track)
        docs.append(vectorizer.transform(tokens))
        track_len = len(tokens)
        if track_len > max_doc_len:
            max_doc_len = track_len


    n_features = len(vectorizer.get_feature_names())
    print(len(docs), "individual tracks loaded with", n_features, "unique notes/chords!\nMaximum document length:", max_doc_len)

    return docs, y, composers, n_features, max_doc_len



if __name__ == "__main__":
    X, y, composers, n_words, max_doc_len = get_docs()
