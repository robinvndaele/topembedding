# Raising warnings
import warnings

# Handling arrays and data.frames
import pandas as pd 
import numpy as np

# Quantitative evaluation with sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import pairwise_distances


def evaluate_embeddings(Ys, labels, model, scoring, params={}, stratify=None, ntimes=10, test_frac=0.1, random_state=None):
    # Obtain test performances over multiple train-test splits
    performances = pd.DataFrame(columns=Ys.keys())
    if random_state is not None:
        np.random.seed(random_state)
    for idx in range(ntimes):
        train, test = train_test_split(range(len(labels)), stratify=stratify, test_size=test_frac)

        # Obtain prediction performance per data embedding
        for key, Y in Ys.items():
            this_train = Y[tuple(np.meshgrid(train, train))] if hasattr(model, "metric") and model.metric == "precomputed" else Y[train,:]
            this_test = Y[tuple(np.meshgrid(train, test))] if hasattr(model, "metric") and model.metric == "precomputed" else Y[test,:]
            this_labels_train = np.array([labels[idx] for idx in train])
            this_labels_test = np.array([labels[idx] for idx in test])
            CV = GridSearchCV(model, params, scoring=scoring)
            CV.fit(this_train, this_labels_train)
            performances.loc["test" + str(idx), key] = CV.score(this_test, this_labels_test)

    return performances
