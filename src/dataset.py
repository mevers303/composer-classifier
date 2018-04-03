# Mark Evers
# Created: 3/30/2018
# model.py
# Functions for getting the data set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.globals import *
from src.midi_archive import get_meta_df
from src.midi_processing import MidiFileText



def get_composers(df):
    composers_df = pd.DataFrame(df.groupby("composer").type.count())
    composers_df.columns = ["works"]

    valid_composers = composers_df[composers_df.works > MINIMUM_WORKS].index.values
    return valid_composers


def get_text(composers, df):
    X_filenames = []
    y_filenames = []
    X_text = []
    y_text = []

    for composer in composers:

        composers_works = df[df.composer == composer]
        if composers_works.composer.count() > MAXIMUM_WORKS:
            composers_works = composers_works.sample(MAXIMUM_WORKS)

        X_filenames.extend(composers_works.index.values)
        y_filenames.extend(composers_works.composer.values)


    total = len(X_filenames)
    i = 0
    for filename, composer in zip(X_filenames, y_filenames):

        text = MidiFileText(filename, df).to_text()
        X_text.extend(text)
        y_text.extend([composer] * len(X_text))

        i += 1
        progress_bar(i, total)


    return X_text, y_text


def tokenize(text):
    return text.split(" ")



def get_docs():

    df = get_meta_df("raw_midi")
    print("All files:   ", df.type.count())
    df = df[df.type == 1]
    print("Type 1 files:", df.type.count())

    composers = get_composers(df)
    X_text, y_text = get_text(composers, df)

    vectorizer = CountVectorizer(tokenizer=tokenize).fit(X_text)

    docs = []
    for track in X_text:
        docs.append(vectorizer.transform(track.split(" ")))

    return docs, y_text, composers, len(vectorizer.get_features)
