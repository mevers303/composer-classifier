# Mark Evers
# 4/12/18
# Results exploring
# eda/results.py

import pickle
import pandas as pd
import matplotlib.pyplot as plt

from file_handlers.dataset import VectorGetterNHot
from model_final import load_from_disk



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


precision_df.plot()
