# Handling arrays
import numpy as np

# Functions for learning in Pytorch
import torch

# MSE loss for PCA
from torch.nn import MSELoss

# Function to preprocess persistence diagrams
from topologylayer.util.process import remove_zero_bars

# Construct scoring functions
from sklearn.metrics import make_scorer

# Working with graphs in Python
import networkx as nx


MACHINE_EPSILON = np.finfo(np.float).eps


MSE = MSELoss()
def pca_loss(X, W, Y=None):
    """
    Pytorch compatible implementation of the ordinary t-SNE loss.
    
    Parameters
    ----------
        X - original high-dimensional data
        W - (approximately) linear projection matrix
        Y - projection of X matrix onto W (computed if missing)
    """
    # Projection of X onto subspace W
    if Y is None: Y = torch.matmul(X, W)
    # Low-rank reconstruction of X
    L = torch.matmul(Y, torch.transpose(W, 0, 1))
    
    # Compute loss
    loss = MSE(L, X)
    
    return loss 


def ortho_loss(W, lambda_W=1):
    """
    Pytorch compatible implementation of loss to encourage orthornormality of a projection matrix.

    Parameters
    ----------
        W - (approximately) linear projection matrix
    """
    # Compute loss
    loss = lambda_W * torch.norm((torch.matmul(torch.transpose(W, 0, 1), W) - torch.eye(2)))

    return loss


def tsne_loss(P, Y):
    """
    Pytorch compatible implementation of the PCA reconstruction loss.

    Parameters
    ----------
        P - high-dimensional neighbor probabilities
        Y - current embedding
    """
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
    """
    Pytorch compatible implementation of the ordinary UMAP loss.

    Parameters
    ----------
        P - high-dimensional neighbor probabilities
        Y - current embedding
        a - UMAP hyperparameter used in pairwise affinities computation
        b - UMAP hyperparameter used in pairwise affinities computation
    """
    # Compute pairwise affinities
    Q = 1 / (1 + a * torch.cdist(Y, Y)**(2 * b))
    Q = torch.max(Q, torch.tensor([MACHINE_EPSILON]))
    oneminQ = torch.max(1 - Q, torch.tensor([MACHINE_EPSILON]))

    # Compute loss
    loss = torch.sum(- P * torch.log(Q) - (1 - P) * torch.log(oneminQ))
    
    return loss


def zero_loss(*args):
    """
    Pytorch compatible implementation zero loss function.
    """
    return torch.tensor(0, dtype=torch.float, requires_grad=True)


def RandomWalk(G, node, t):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------

    """
    walk = [node] # Walk starts from this node
    for i in range(t - 1):      
        if not nx.is_weighted(G):
            W = np.ones(len(G[node]))
        else: 
            W = [G[node][n]["weight"] for n in G.neighbors(node)]
        node = np.random.choice(list(G.neighbors(node)), p=W / np.sum(W))            
        walk.append(node)

    return walk


def deepwalk_loss(model, G, w, t):
    """
    Pytorch compatible implementation of the deepwalk loss.
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------

    """
    loss = torch.tensor(0).type(torch.float)
    for vi in list(G.nodes()):
        wvi = RandomWalk(G, vi, t)
        for j in range(len(wvi)):
            for k in range(max(0, j - w) , min(j + w, len(wvi))):
                prob = model(wvi[j], wvi[k])
                loss = loss - torch.log(prob)

    return loss


class DiagramLoss(torch.nn.Module):
    """
    Applies function g over points in a diagram sorted by persistence.
    Original source: https://github.com/bruel-gabrielsson/TopologyLayer

    Parameters
    ----------
        dim - homology dimension to work over
        g - pytorch compatible function to evaluate on each diagram point
        i - start of summation over ordered diagram points
        j - end of summation over ordered diagram points
        remove_zero = Flag to remove zero-length bars (default=True)
    """
    def __init__(self, dim, g, i=1, j=np.inf, remove_zero=True):
        super(DiagramLoss, self).__init__()
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
