# Mark Evers
# 4/12/18
# Results exploring
# eda/results.py

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# from file_handlers.dataset import VectorGetterNHot
# from model_final import load_from_disk





def plot_result(df, title, xlabel, ylabel):

    df = df.sort_values(ascending=False)
    ax = df.plot(kind="bar", width=.75)

    cmap = plt.get_cmap("gist_rainbow")
    for result, bar in zip(df.values, ax.get_children()[:18]):
        bar.set_color(cmap(result - .39))

    plt.xticks(rotation=55)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
        tick.set_fontsize(18)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)

    plt.title(title, fontsize=32)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.ylim(ymin=0, ymax=1)
    plt.tight_layout()
    plt.legend().remove()
    plt.show()



def plot_both_results(recall_df, precision_df):

    n = len(dataset.composers)
    ind = np.arange(n)  # the x locations for the groups
    width = 0.4  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(alpha=.25)

    rec_vals = recall_df.iloc[-1]
    bar1 = ax.bar(ind + width * .5, rec_vals, width, color='dodgerblue')
    prec_vals = precision_df.iloc[-1].values
    bar2 = ax.bar(ind + width * 1.5, prec_vals, width, color='rebeccapurple')

    ax.set_ylabel('Scores')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(dataset.composers)
    ax.legend((bar1[0], bar2[0]), ('Recall', 'Precision'))

    # def autolabel(rects):
    #     for rect in rects:
    #         h = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, "{:.2f}".format(h).replace("0.", "."),
    #                 ha='center', va='bottom')
    #
    # autolabel(bar1)
    # autolabel(bar2)


    plt.xticks(rotation=55)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
        tick.set_fontsize(18)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_fontsize(18)

    plt.title("Categorical Metrics", fontsize=32)
    plt.xlabel("Composer", fontsize=22)
    plt.ylabel("Value", fontsize=22)
    # plt.ylim(ymin=0, ymax=1)
    plt.tight_layout()
    plt.show()


with open("midi/classical/dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

with open("models/final_metrics.pkl", "rb") as f:
    results = pickle.load(f)


accuracy_results = []
precision_results = []
recall_results = []
fscore_results = []

for accuracy, precision, recall, fscore in results:

    accuracy_results.append(accuracy)
    precision_results.append(precision)
    recall_results.append(recall)
    fscore_results.append(fscore)


precision_df = pd.DataFrame(precision_results, columns=dataset.composers)
recall_df = pd.DataFrame(recall_results, columns=dataset.composers)
fscore_df = pd.DataFrame(fscore_results, columns=dataset.composers)


plot_both_results(recall_df, precision_df)
