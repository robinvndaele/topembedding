# Handling arrays and data frames
import numpy as np
import pandas as pd

# Tracking computation times
import time

# Setting seeds for reproducibility
import random

# Working with graphs in Python
import networkx as nx

# Functions for learning in Pytorch
import torch
import pytorch_lightning as pl
from torch import sigmoid
from torch.nn import Embedding
from torch.nn.functional import binary_cross_entropy
from sklearn.metrics import roc_auc_score

# Embedding and topological loss functions
from Code.losses import pca_loss, ortho_loss, umap_loss, tsne_loss, zero_loss, deepwalk_loss

# Functions to initialize embeddings
from sklearn.decomposition import PCA as skPCA
from Code.embedding_init import tsne_init, umap_init

# Helper functions for network embedding
from Code.splitter import compute_tr_val_split, construct_adj_matrix
from Code.dataloader import config_network_loader, config_edge_loader


def PCA(X, dim=2, emb_loss=True, top_loss=zero_loss, lambda_W=1e4, num_epochs=250, learning_rate=1e-3, eps=1e-07, random_state=None):
    """
    Conduct topologically regularized PCA embedding.
    
    Parameters
    ----------
        X - high-dimensional data matrix
        dim - required dimensionality of the embedding
        emb_loss - whether to use the PCA reconstruction loss function, if false, the zero loss function is used insteads
        top_loss - topological loss function (= prior) for topological regularization
        lambda_W - regularization hyperparameter to encourage orthonormality of the projection matrix
        num_epochs - the number of epochs of the optimization
        learning_rate - learning rate of the Adam optimizer
        eps - term added to the denominator to improve numerical stability of the Adam optimizer
        random_state - used to set random seed for reproducibility

    Returns
    -------
    Y - data embedding of X
    W - optimized (approximately) linear projection matrix
    losses - data frame to investigate the losses according to epochs
    """
    # Track total embedding time
    start_time = time.time()   
 
    # Center the data
    X = X - X.mean(axis=0)
    
    # Initialize the embedding with PCA
    pca = skPCA(n_components=dim, random_state=random_state).fit(X)
    W = pca.components_.transpose()
    if top_loss.__name__ == "zero_loss":
        Y = pca.transform(X)
        elapsed_time = time.time() - start_time
        print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        return (Y, W)
    W = torch.tensor(W).type(torch.float)
    W = torch.autograd.Variable(W, requires_grad=True)
    
    # Store losses for potential further exploration
    losses = np.zeros([num_epochs, 3])
    
    # Initialize the optimization
    if not random_state is None:
        random.seed(random_state)
        torch.manual_seed(random_state)
    X = torch.tensor(X).type(torch.float)  
    optimizer = torch.optim.Adam([W], lr=learning_rate, eps=eps)
    
    # Conduct the optimization
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute projection of X onto subspace W
        Y = torch.matmul(X, W)

        # Compute the losses
        loss_pca = pca_loss(X, W, Y) if emb_loss else zero_loss()
        loss_W = ortho_loss(W, lambda_W)
        loss_top = top_loss(Y)
        loss = loss_pca + loss_W + loss_top

        # Store the losses
        losses[epoch,:] = [loss_pca.item(), loss_W.item(), loss_top.item()]

        # Conduct optimization step
        loss.backward()
        optimizer.step()

        # Print losses according to epoch        
        if epoch == 0 or (epoch + 1) % (int(num_epochs / 10)) == 0:
            print ("[epoch %d] [emb. loss: %f, ortho. loss: %f, top. loss: %f, total loss: %f]" % 
                   (epoch + 1, loss_pca, loss_W, loss_top, loss))
    
    # Obtain numpy embedding matrix
    W = W.detach().numpy()
    Y = np.dot(X.numpy(), W)

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Construct data frame storing our losses
    losses = pd.DataFrame(losses, columns=["embedding", "orthonormality", "topological"])
    
    return (Y, W, losses)


def TSNE(X, dim=2, emb_loss=True, top_loss=zero_loss, initial_components=30, perplexity=30, num_epochs=250, learning_rate=1e-3, eps=1e-07, random_state=None):
    """
    Conduct topologically regularized t-SNE embedding.

    Parameters
    ----------
        X - high-dimensional data matrix
        dim - required dimensionality of the embedding
        emb_loss - whether to use the UMAP loss function, if false, the zero loss function is used insteads
        top_loss - topological loss function (= prior) for topological regularization
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        perplexity - desired number of nearest neighbors
        num_epochs - the number of epochs of the optimization
        learning_rate - learning rate of the Adam optimizer
        eps - term added to the denominator to improve numerical stability of the Adam optimizer
        random_state - used to set random seed for reproducibility

    Returns
    -------
    Y - data embedding of X
    losses - data frame to investigate the losses according to epochs
    """
    # Track total embedding time
    start_time = time.time()   
    
    # Initialize t-SNE embedding with PCA
    P, Y = tsne_init(X, n_components=dim, initial_components=initial_components, perplexity=perplexity, random_state=random_state)
    
    # Store losses for potential further exploration
    losses = np.zeros([num_epochs, 2])
    
    # Initialize the optimization
    if not random_state is None:
        random.seed(random_state)
        torch.manual_seed(random_state)
    Y = torch.autograd.Variable(Y, requires_grad=True)
    optimizer = torch.optim.Adam([Y], lr=learning_rate, eps=eps)
    
    # Conduct the optimization
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute the losses
        loss_tsne = tsne_loss(P, Y) if emb_loss else zero_loss()
        loss_top = top_loss(Y)
        loss = loss_tsne + loss_top

        # Store the losses
        losses[epoch,:] = [loss_tsne.item(), loss_top.item()]

        # Conduct optimization step
        loss.backward()
        optimizer.step()

        # Recenter embedding
        Y - torch.mean(Y, 0)

        # Print losses according to epoch        
        if epoch == 0 or (epoch + 1) % (int(num_epochs / 10)) == 0:
            print ("[epoch %d] [emb. loss: %f, top. loss: %f, total loss: %f]" % (epoch + 1, loss_tsne, loss_top, loss))
    
    # Obtain numpy embedding matrix
    Y = Y.detach().numpy()

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Construct data frame storing our losses as well as the variables from which they are computed
    losses = {"losses":pd.DataFrame(losses, columns=["embedding", "topological"]), "P":P}
    
    return (Y, losses)


def UMAP(X, dim=2, emb_loss=True, top_loss=zero_loss, initial_components=30, n_neighbors=15, spread=1.0, min_dist=0.1, num_epochs=250, learning_rate=1e-3, eps=1e-07, random_state=None):
    """
    Conduct topologically regularized UMAP embedding.

    Parameters
    ----------
        X - high-dimensional data matrix
        dim - required dimensionality of the embedding
        emb_loss - whether to use the UMAP loss function, if false, the zero loss function is used insteads
        top_loss - topological loss function (= prior) for topological regularization
        initial_components - dimensionality of the PCA projection space in which the neighbor probabilities are computed
        n_neighbors - desired number of nearest neighbors
        spread - hyperparameter to control inter-cluster distance
        min_dist - hyperparameter to control cluster size
        num_epochs - the number of epochs of the optimization
        learning_rate - learning rate of the Adam optimizer
        eps - term added to the denominator to improve numerical stability of the Adam optimizer
        random_state - used to set random seed for reproducibility

    Returns
    -------
    Y - data embedding of X
    losses - data frame to investigate the losses according to epochs
    """
    # Track total embedding time
    start_time = time.time()   
    
    # Initialize UMAP embedding with PCA
    P, Y, a, b = umap_init(X, n_components=dim, initial_components=initial_components, n_neighbors=n_neighbors, spread=spread, min_dist=min_dist, random_state=random_state) # numba code compilation now completed
    if(num_epochs < 1): return # can be used to break after numba code compilation
    
    # Store losses for potential further exploration
    losses = np.zeros([num_epochs, 2])
    
    # Initialize the optimization
    if not random_state is None:
        random.seed(random_state)
        torch.manual_seed(random_state)
    Y = torch.autograd.Variable(Y, requires_grad=True)
    optimizer = torch.optim.Adam([Y], lr=learning_rate, eps=eps)
    
    # Conduct the optimization
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute the losses
        loss_umap = umap_loss(P, Y, a, b) if emb_loss else zero_loss()
        loss_top = top_loss(Y)
        loss = loss_umap + loss_top

        # Store the losses
        losses[epoch,:] = [loss_umap.item(), loss_top.item()]

        # Conduct optimization step
        loss.backward()
        optimizer.step()

        # Recenter embedding
        Y - torch.mean(Y, 0)

        # Print losses according to epoch        
        if epoch == 0 or (epoch + 1) % (int(num_epochs / 10)) == 0:
            print ("[epoch %d] [emb. loss: %f, top. loss: %f, total loss: %f]" % (epoch + 1, loss_umap, loss_top, loss))
    
    # Obtain numpy embedding matrix
    Y = Y.detach().numpy()

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Construct dictionary storing our losses as well as the variables from which they are computed
    losses = {"losses":pd.DataFrame(losses, columns=["embedding", "topological"]), "P":P, "a":a, "b":b}
    
    return (Y, losses)


class GraphInProdEmbeddingModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(GraphInProdEmbeddingModel, self).__init__()
        self.name = "GraphEmbedding"
        self.n = kwargs["n"]
        self.dim = kwargs["dim"]
        self.learning_rate = kwargs["learning_rate"]
        self.eps = kwargs["eps"]
        self.emb_loss = kwargs["emb_loss"]
        self.top_loss = kwargs["top_loss"]
        self.optimizer = getattr(torch.optim, "Adam")

        self.embedding = Embedding(self.n, self.dim)
        self.embedding.weight.data.normal_(0, 0.1)

        self.b_node = torch.nn.parameter.Parameter(torch.Tensor(self.n))
        torch.nn.init.normal_(self.b_node, std=0.1)

        self.b = torch.nn.parameter.Parameter(torch.Tensor(1))
        torch.nn.init.normal_(self.b, std=0.1)

    def forward(self, uids, iids):
        return sigmoid((self.embedding(uids) * self.embedding(iids)).sum(1) + self.b_node[uids] + self.b_node[iids] + self.b)
    
    def training_step(self, batch, batch_idx):
        uids, iids, target = batch        
        pred = self(uids, iids)
        loss_emb = binary_cross_entropy(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum() if self.emb_loss else zero_loss()
        loss_top = self.top_loss(self.embedding)
        loss = loss_emb + loss_top

        # self.log("train_loss", loss, on_epoch=True)
        self.log("emb. loss", loss_emb, prog_bar=True)
        self.log("top. loss", loss_top, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        uids, iids, target = batch
        pred = self(uids, iids)
        auc = roc_auc_score(target, pred)
        self.log(f"val_auc_{loader_idx}", auc, prog_bar=True)

    def configure_optimizers(self):
        print(f"Config optimizer with learning rate {self.learning_rate}")
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate, eps=self.eps)
        return optimizer


def GraphInProdEmbed(G, dim=2, emb_loss=True, top_loss=zero_loss, train_frac=0.9, num_epochs=250, learning_rate=1e-1, eps=1e-07, random_state=None):

    # Track total embedding time
    start_time = time.time()   

    # Prepare the data for training
    if not random_state is None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    E = np.array(G.edges())
    A = construct_adj_matrix(E, E.max() + 1)
    tr_A, tr_E, val_E = compute_tr_val_split(A, train_frac)
    tr_A_loader = config_network_loader(tr_A)
    tr_E_loader = config_edge_loader(tr_E)
    val_E_loader = config_edge_loader(val_E)

    # Conduct the training
    trainer = pl.Trainer(num_sanity_val_steps=0, checkpoint_callback=False, logger=False, max_epochs=num_epochs)
    model = GraphInProdEmbeddingModel(n=tr_A.shape[0], dim=dim, emb_loss=emb_loss, top_loss=top_loss, learning_rate=learning_rate, eps=eps)
    trainer.fit(model, tr_A_loader, val_dataloaders=[tr_E_loader, val_E_loader])

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Return the embedded graph
    return model


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
        w //= 2

    return count


def func_n(w, j):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """
    li=[w]
    while(w != 1):
        w = w // 2
        li.append(w)

    li.reverse()
    
    return li[j]



class HierarchicalModel(torch.nn.Module):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """    
    def __init__(self, size_vertex, dim, init=None):
        super(HierarchicalModel, self).__init__()
        self.size_vertex = size_vertex
        self.phi = torch.nn.Parameter(torch.rand((size_vertex, dim)) if init is None else torch.tensor(init), requires_grad=True) 
        self.prob_tensor = torch.nn.Parameter(torch.rand((2 * size_vertex, dim), requires_grad=True))
    
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
        
            p = p * torch.sigmoid(mult * torch.matmul(self.prob_tensor[func_n(w, j)], h))
        
        return p


def DeepWalk(G, dim=2, emb_loss=True, top_loss=zero_loss, init=None, num_epochs=250, learning_rate=1e-2, w=3, t=6, random_state=None):
    """
    Original source: https://github.com/dsgiitr/graph_nets

    Parameters
    ----------


    Returns
    -------

    """
    # Track total embedding time
    start_time = time.time()   
    
    if not random_state is None:
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    model = HierarchicalModel(size_vertex=len(G.nodes()), dim=dim, init=init)
    
    for epoch in range(num_epochs):
        loss_emb = deepwalk_loss(model, G, w, t) if emb_loss else zero_loss()
        loss_top = top_loss(model.phi)
        loss = loss_emb + loss_top
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                param.data.sub_(learning_rate * param.grad)
                param.grad.data.zero_()

        # Print losses according to epoch        
        if epoch == 0 or (epoch + 1) % (int(num_epochs / 10)) == 0:
            print ("[epoch %d] [emb. loss: %f, top. loss: %f, total loss: %f]" % (epoch + 1, loss_emb, loss_top, loss))

    # Print total embedding time
    elapsed_time = time.time() - start_time
    print("Time for embedding: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        
    return model
