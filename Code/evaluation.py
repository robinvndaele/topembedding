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


def angular_scorer_func(ground_truth, prediction):
    new_prediction = prediction % (2 * np.pi)
    diff_for_k = np.array([np.abs(new_prediction + k * 2 * np.pi - ground_truth) for k in [-1, 0, 1]]).transpose()
    new_prediction = new_prediction + [(np.argmin(d) - 1) * 2 * np.pi for d in diff_for_k]
    return r2_score(ground_truth, new_prediction)
angular_scorer = make_scorer(angular_scorer_func, greater_is_better=True)


def angular_distances(Y):
    # Get angular coordinate of each point in Y
    angles = np.arctan2(Y[:,1], Y[:,0])

    # Get pairwise angular distances between points in Y
    angularD = np.zeros([Y.shape[0], Y.shape[0]])
    for idx1 in np.arange(Y.shape[0] - 1):
        for idx2 in np.arange(idx1 + 1, Y.shape[0]):
            angularD[idx1, idx2] = np.abs(np.arctan2(np.sin(angles[idx1] - angles[idx2]), np.cos(angles[idx1] - angles[idx2])))
            angularD[idx2, idx1] = angularD[idx1, idx2]

    return angularD


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
            CV = GridSearchCV(model, params, scoring=angular_scorer if scoring=="angular" else scoring)
            CV.fit(this_train, this_labels_train)
            performances.loc["test" + str(idx), key] = CV.score(this_test, this_labels_test)

    return performances
