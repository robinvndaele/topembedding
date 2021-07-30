# Handling arrays
import numpy as np

# Working with graphs in Python
import networkx as nx 

# Functions for deep learning (Pytorch)
import torch
from torch import nn

# Function to preprocess persistence diagrams
from topologylayer.util.process import remove_zero_bars

# Functions to initialize dimensionality reductions
from sklearn.decomposition import PCA
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit


MACHINE_EPSILON = np.finfo(np.float).eps


def prob_high_dim(dist, rho, sigma, dist_row):
    """
    For each row of Euclidean distance matrix (dist_row) compute
    probability in high dimensions (1D array)
    """
    d = dist[dist_row] - rho[dist_row]
    d[d < 0] = 0
    return np.exp(- d / sigma)


def k(prob):
    """
    Compute n_neighbors = k (scalar) for each 1D array of high-dimensional probability
    """
    return np.power(2, np.sum(prob))


def sigma_binary_search(k_of_sigma, fixed_k):
    """
    Solve equation k_of_sigma(sigma) = fixed_k 
    with respect to sigma by the binary search algorithm
    """
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        if k_of_sigma(approx_sigma) < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5:
            break
    return approx_sigma


def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    
    return params[0], params[1]


def tsne_initialize(X, n_components=2, initial_components=30, perplexity=30, random_state=None):
    
    X_pca = PCA(n_components=initial_components, random_state=random_state).fit_transform(X)
    P = _joint_probabilities(distances=pairwise_distances(X_pca, squared=True), desired_perplexity=perplexity, verbose=0)
    P = torch.max(torch.tensor(squareform(P)).type(torch.float), torch.tensor([MACHINE_EPSILON]))
    
    return P, torch.tensor(X_pca[:,range(n_components)]).type(torch.float)


def umap_initialize(X, n_components=2, initial_components=30, n_neighbors=15, spread=1.0, min_dist=0.1, random_state=None):

    X_pca = PCA(n_components=initial_components, random_state=random_state).fit_transform(X)
    dist = pairwise_distances(X_pca, squared=True)
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    P = np.zeros((X.shape[0], X.shape[0])) 
    sigma_array = []
    
    for dist_row in range(X.shape[0]):
        
        func = lambda sigma: k(prob_high_dim(dist, rho, sigma, dist_row))
        binary_search_result = sigma_binary_search(func, n_neighbors)
        P[dist_row] = prob_high_dim(dist, rho, binary_search_result, dist_row)
        sigma_array.append(binary_search_result)
        
    P = P + np.transpose(P) - np.multiply(P, np.transpose(P))
    P = torch.max(torch.tensor(P).type(torch.float), torch.tensor([MACHINE_EPSILON]))
    
    a, b = find_ab_params(spread, min_dist)
    
    return P, torch.tensor(X_pca[:,range(n_components)]).type(torch.float), a, b


def mds_initialize(X, n_components=2, initial_components=30, random_state=None):
    
    X_pca = torch.tensor(PCA(n_components=initial_components, random_state=random_state).fit_transform(X)).type(torch.float)
    D = torch.cdist(X_pca, X_pca)
    
    return D, X_pca[:,range(n_components)]


def tsne_loss(P, Y):
    
    # Compute pairwise affinities
    sum_Y = torch.sum(Y * Y, 1)
    num = -2 * torch.mm(Y, Y.t())
    num = 1 / (1 + torch.add(torch.add(num, sum_Y).t(), sum_Y))
    num[range(Y.shape[0]), range(Y.shape[0])] = 0
    Q = num / torch.sum(num)
    Q = torch.max(Q, torch.tensor([MACHINE_EPSILON]))
    
    # Compute loss
    loss = torch.sum(P * torch.log(P / Q))
    
    return loss 


def umap_loss(P, Y, a, b):
    
    # Compute pairwise affinities
    Q = 1 / (1 + a * torch.cdist(Y, Y)**(2 * b))
    Q = torch.max(Q, torch.tensor([MACHINE_EPSILON]))
    oneminQ = torch.max(1 - Q, torch.tensor([MACHINE_EPSILON]))

    # Compute loss
    loss = torch.sum(- P * torch.log(Q) - (1 - P) * torch.log(oneminQ))
    
    return loss


def mds_loss(D, Y):
    
    # Compute pairwise distances
    D_low = torch.cdist(Y, Y)

    # Compute loss
    loss = torch.mean((D - D_low)**2)
    
    return loss


class DiagramFeature(torch.nn.Module):
    """
    applies function g over points in a diagram sorted by persistence
    parameters:
        dim - homology dimension to work over
        g - pytorch compatible function to evaluate on each diagram point
        i - start of summation over ordered diagram points
        j - end of summation over ordered diagram points
        remove_zero = Flag to remove zero-length bars (default=True)
    """
    def __init__(self, dim, g, i=1, j=np.inf, remove_zero=True):
        super(DiagramFeature, self).__init__()
        self.dim = dim
        self.g = g
        self.i = i - 1
        self.j = j
        self.remove_zero = remove_zero

    def forward(self, dgminfo):
        dgms, issublevel = dgminfo
        dgm = dgms[self.dim]
        if self.remove_zero:
            dgm = remove_zero_bars(dgm)
        
        lengths = dgm[:,1] - dgm[:,0]
        indl = torch.argsort(lengths, descending=True)
        dgm = dgm[indl[self.i:min(dgm.shape[0], self.j)]]

        loss = torch.sum(torch.stack([self.g(dgm[i]) for i in range(dgm.shape[0])], dim=0))

        return loss


def RandomWalk(G, node, t):
    walk = [node] # Walk starts from this node
    
    for i in range(t - 1):
        
        if not nx.is_weighted(G):
            W = np.ones(len(G[node]))
        else: 
            W = [G[node][n]["weight"] for n in G.neighbors(node)]
        node = np.random.choice(list(G.neighbors(node)), p=W / np.sum(W))
            
        walk.append(node)

    return walk


def func_L(w):
    """
    Parameters
    ----------
    w: Leaf node.
    
    Returns
    -------
    count: The length of path from the root node to the given vertex.
    """
    count = 1
    while(w != 1):
        count += 1
        w //=2

    return count


def func_n(w, j):
    li=[w]
    while(w != 1):
        w = w // 2
        li.append(w)

    li.reverse()
    
    return li[j]


def sigmoid(x):
    out = 1 / (1 + torch.exp(-x))
    return out


class HierarchicalModel(torch.nn.Module):
    
    def __init__(self, size_vertex, dim):
        super(HierarchicalModel, self).__init__()
        self.size_vertex = size_vertex
        self.phi = nn.Parameter(torch.rand((size_vertex, dim), requires_grad=True))   
        self.prob_tensor = nn.Parameter(torch.rand((2 * size_vertex, dim), requires_grad=True))
    
    def forward(self, wi, wo):
        one_hot = torch.zeros(self.size_vertex)
        one_hot[wi] = 1
        w = self.size_vertex + wo
        h = torch.matmul(one_hot, self.phi)
        p = torch.tensor([1.0])
        for j in range(1, func_L(w) - 1):
            mult = -1
            if(func_n(w, j + 1) == 2 * func_n(w, j)): # Left child
                mult = 1
        
            p = p * sigmoid(mult * torch.matmul(self.prob_tensor[func_n(w, j)], h))
        
        return p
