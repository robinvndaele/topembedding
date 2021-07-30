import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from Experiments.pytorch.splitter import compute_tr_val_split, construct_adj_matrix
from Experiments.pytorch.dataloader import config_network_loader, config_edge_loader
from Experiments.pytorch.cne import ConditionalNetworkEmbedding
from Experiments.pytorch.topemb import TopEmbedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset')    
    parser.add_argument('--method', type=str, required=True, help='embedding method')
    parser.add_argument('--top_type', type=str, help='topology type')    
    args = parser.parse_args()
    
    # prepare data
    if args.dataset == "harry_potter":
        E = pd.read_csv("./Data/Harry Potter/relations.csv")[["source", "target"]].values
    elif args.dataset == "facebook":
        E = np.loadtxt("./Data/facebook_combined.txt", delimiter=" ", dtype=int)
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")        
    A = construct_adj_matrix(E, E.max() + 1)
    tr_A, tr_E, val_E = compute_tr_val_split(A, 0.9)
    tr_A_loader = config_network_loader(tr_A)
    tr_E_loader = config_edge_loader(tr_E)
    val_E_loader = config_edge_loader(val_E)

    # train
    trainer = pl.Trainer(num_sanity_val_steps=0, checkpoint_callback=False, logger=False, max_epochs=100)
    if args.method == "cne":
        model = ConditionalNetworkEmbedding(tr_A=tr_A, d=2, learning_rate=0.1)
    elif args.method == "top":
        model = TopEmbedding(tr_A=tr_A, d=2, learning_rate=0.1, top_type=args.top_type, top_frac=1, lambda_top=-0.1)
    else:
        raise ValueError(f"unknown embedding mehtod: {args.method}")
    trainer.fit(model, tr_A_loader, val_dataloaders=[tr_E_loader, val_E_loader])

    # plot
    fig, ax = plt.subplots(figsize=(16, 10))    
    plt.scatter(model.embedding.weight.data[:,0], model.embedding.weight.data[:,1])
    plt.title(f"{args.dataset} {args.method} embedding")
    if args.method == "top":
        plt.savefig(f"{args.dataset}_{args.method}_{args.top_type}.png")
    else:
        plt.savefig(f"{args.dataset}_{args.method}.png")