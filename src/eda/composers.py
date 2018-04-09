# Mark Evers
# Created: 3/30/2018
# composers.py
# Let's look at the composers!

import pandas as pd
import os

from src.globals import MINIMUM_WORKS


def get_df(base_dir="midi/classical"):

    df = pd.read_csv(os.path.join(base_dir, "meta.csv"), index_col="filename")
    df = df[df.type == 1]
    return df


def get_composer_works(df):
    """
    Returns a list of composers that have at least MINIMUM_WORKS pieces.

    :return: A list of composers
    """

    # valid_composers = ["Bach", "Mozart", "Beethoven", "Tchaikovsky", "Vivaldi", "Chopin", "Debussy", "Schubert", "Stravinsky"]

    composers_df = pd.DataFrame(df.groupby("composer").type.count())
    composers_df.columns = ["works"]

    return composers_df


def get_valid_composers(composers_df):

    valid_composers = set(composers_df[composers_df.works > MINIMUM_WORKS].index.values)
    valid_composers.update(["Bach", "Mozart", "Beethoven", "Tchaikovsky", "Vivaldi", "Chopin", "Debussy", "Schubert", "Stravinsky"])

    print("Found", len(valid_composers), "composers:", ", ".join(valid_composers))
    print(composers_df[composers_df.index.isin(valid_composers)])

    return valid_composers


if __name__ == "__main__":

    df = get_df()
    composers_df = get_composer_works(df)
    valid_composers = get_valid_composers(composers_df)
