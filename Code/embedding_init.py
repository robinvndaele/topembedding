# Handling arrays
import numpy as np

# Functions for learning in Pytorch
import torch

# General initialization functions
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import pairwise_distances

# Functions to initialize t-SNE
from sklearn.manifold._t_sne import _joint_probabilities
from scipy.spatial.distance import squareform

# Functions to initialize UMAP
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set


MACHINE_EPSILON = np.finfo(np.float).eps


def tsne_init(X, n_components, initial_components, perplexity, random_state=None):
    """
    Initialize t-SNE embedding with PCA.

    Parameters
    ----------
        X - high-dimensional data matrix
        n_components - required dimensionality of the embedding
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        perplexity - guess about the number of close neighbors each point has
        random_state - used to set random seed for reproducibility
    """
    if initial_components < X.shape[1]:
        X = skPCA(n_components=initial_components, random_state=random_state).fit_transform(X)
    P = _joint_probabilities(distances=pairwise_distances(X, squared=True), desired_perplexity=perplexity, verbose=0)
    P = torch.max(torch.tensor(squareform(P)).type(torch.float), torch.tensor([MACHINE_EPSILON]))
    
    return P, torch.tensor(X[:,range(n_components)]).type(torch.float)


def umap_init(X, n_components, initial_components, n_neighbors, spread, min_dist, random_state=None):
    """
    Initialize UMAP embedding with PCA.

    Parameters
    ----------
        X - high-dimensional data matrix
        n_components - required dimensionality of the embedding
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        n_neighbors - desired number of nearest neighbors
        spread - hyperparameter to control inter-cluster distance
        min_dist - hyperparameter to control cluster size
        random_state - used to set random seed for reproducibility
    """
    if initial_components < X.shape[1]:
        X = skPCA(n_components=initial_components, random_state=random_state).fit_transform(X)
    dist = pairwise_distances(X, metric="euclidean")

    a, b = find_ab_params(spread, min_dist)
    P = fuzzy_simplicial_set(dist, n_neighbors, random_state=random_state, metric="precomputed")[0].tocoo()
    P = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((P.row, P.col))), torch.FloatTensor(P.data), torch.Size(P.shape)).to_dense()
    P = torch.max(P, torch.tensor([MACHINE_EPSILON]))
    
    return P, torch.tensor(X[:,range(n_components)]).type(torch.float), a, b
