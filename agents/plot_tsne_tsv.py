from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import csv
import numpy as np


input_tsv = 'outcomes.tsv'
labels_tsv = 'labels.txv'
with open(input_tsv, 'r') as infile:
    outcome_reader = csv.load(infile, delimiter='\t')
    with open(labels_tsv, 'r') as labelfile:
        label_reader = csv.load(labelfile, delimiter='\t')
        outcomes = []
        labels = []
        unique_labels = set()
        outcome_append = outcomes.append
        label_append = labels.append
        for outcome in outcome_reader:
            outcome_append(outcome)
            next_label = label_reader.next()
            unique_labels.add(next_label)
            label_append(next_label)
        tsne = TSNE(n_components=2, verbose=True)
        embeddings = tsne.fit_transform(outcomes)
        assert len(embeddings) == len(labels)
        scatters = {}
        for ulabel in list(unique_labels):
            scatters[ulabel] = [embeddings[i] for i, label in enumerate(labels) if label == ulabel]
        for label,data in scatters:
            e_arr = np.array(data)
            assert e_arr.shape[1] == 2
            x = e_arr[:, 0]
            y = e_arr[:, 1]
            plt.scatter(x, y, label=label)
        plt.legend()
        plt.show()




