# Mark Evers
# Created: 3/30/2018
# composers.py
# Let's look at the composers!

import pandas as pd
import os
import matplotlib.pyplot as plt

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

    # valid_composers = set(composers_df[composers_df.works > MINIMUM_WORKS].index.values)
    # valid_composers.update(["Bach", "Mozart", "Beethoven", "Tchaikovsky", "Vivaldi", "Chopin", "Debussy", "Schubert", "Stravinsky"])

    valid_composers = ["Bach", "Beethoven", "Chopin", "Debussy", "Giuliani", "Handel", "Hays", "Hewitt", "Mozart",
                       "Paganini", "Scarlatti", "Schubert", "Sor", "Tchaikovsky", "Thomas", "Tucker", "Vivaldi",
                       "Webster"]

    print("Found", len(valid_composers), "composers:", ", ".join(valid_composers))
    # print(composers_df[composers_df.index.isin(valid_composers)])

    return valid_composers


def get_valid_composers_df():

    df = get_df()
    composers_df = get_composer_works(df)
    valid_composers = get_valid_composers(composers_df)

    df = composers_df[composers_df.index.isin(valid_composers)].sort_values(by="works", ascending=False)

    for i in df.index:
        if df.loc[i].works > 120:
            df.loc[i].works = 120


    return df


def plot_balance(composers_df, valid_composers):

    df = composers_df[composers_df.index.isin(valid_composers)].sort_values(by="works", ascending=False)

    for i in df.index:
        if df.loc[i].works > 120:
            df.loc[i].works = 120


    # df.plot.pie("works")
    ax = df.plot(kind="bar", width=.75)
    cmap = plt.get_cmap("nipy_spectral_r")
    for works, bar in zip(df.works, ax.get_children()[:18]):
        bar.set_color(cmap(works))
    plt.xticks(rotation=55)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
        tick.set_fontsize(18)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)
    plt.title("Class balance", fontsize=32)
    plt.xlabel("Composer", fontsize=22)
    plt.ylabel("Number of works", fontsize=22)
    plt.tight_layout()
    plt.legend().remove()
    plt.show()


def get_random_guess_prob():

    df = get_valid_composers_df()

    total = df.works.sum()
    n = df.works.count()
    prob = 0

    for x in df.works.values:
        prob += (x / total) * (1 / n)
        print(prob)

    return prob


if __name__ == "__main__":

    df = get_df()
    df = get_composer_works(df)
