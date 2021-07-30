import math

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from torch import DoubleTensor, LongTensor, sigmoid
from torch._C import Value
from torch.nn import Embedding
from torch.nn.functional import binary_cross_entropy


class ConditionalNetworkEmbedding(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ConditionalNetworkEmbedding, self).__init__()
        self.name = "ConditionalNetworkEmbedding"
        A, d = kwargs["tr_A"],kwargs["d"]
        self.learning_rate = kwargs["learning_rate"]
        self.n = A.shape[0]
        self.model_params = kwargs
        self.optimizer = getattr(torch.optim, "Adam")

        self.embedding =Embedding(self.n, d)
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
        loss = binary_cross_entropy(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx):
        uids, iids, target = batch
        pred = self(uids, iids)
        auc = roc_auc_score(target, pred)
        self.log(f"val_auc_{loader_idx}", auc, prog_bar=True)


    def configure_optimizers(self):
        print(f"Config optimizer with learning rate {self.learning_rate}")
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer