import random

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from torch import DoubleTensor, LongTensor, sigmoid
from torch.nn import Embedding
from torch.nn.functional import binary_cross_entropy


from topologylayer.nn import AlphaLayer
from topembed import DiagramFeature


class TopEmbedding(pl.LightningModule):
    def __init__(self, **kwargs):
        super(TopEmbedding, self).__init__()
        self.name = "TopEmbedding"
        A, d = kwargs["tr_A"], kwargs["d"]
        self.learning_rate = kwargs["learning_rate"]
        self.top_type, self.top_frac, self.lambda_top = kwargs["top_type"], kwargs["top_frac"], kwargs["lambda_top"]
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


    def top_criterion(self, embedding, top_type, n, top_frac, lambda_top):
        g = lambda p: p[1] if p[1] < np.inf else torch.tensor(0).type(torch.float)
        I = LongTensor(random.sample(range(n), int(range(n) * top_frac)) if top_frac < 1 else range(n))
        
        if top_type == "circular":
            top_layer = AlphaLayer(maxdim=1) # alpha complex layer            
            persistence_feature = DiagramFeature(dim=1, j=1, g=g)            
        elif top_type == "comp2":
            top_layer = AlphaLayer(maxdim=0)            
            persistence_feature = DiagramFeature(dim=0, j=2, g=g) # compute persistence of second most prominent gap
        else:
            raise ValueError(f"undefined topology type: {top_type}")
        
        dgminfo = top_layer(embedding(I))                    
        top_loss = lambda_top * persistence_feature(dgminfo)
        return top_loss
    
    def training_step(self, batch, batch_idx):
        uids, iids, target = batch        
        pred = self(uids, iids)
        emb_loss = binary_cross_entropy(pred.view(-1, 1).float(), target.view(-1, 1).float()).sum()        
        top_loss = self.top_criterion(self.embedding, self.top_type, self.n, self.top_frac, self.lambda_top)
        loss = emb_loss + top_loss

        # self.log("train_loss", loss, on_epoch=True)
        self.log("emb_loss", emb_loss, prog_bar=True)
        self.log("top_loss", top_loss, prog_bar=True)
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

    
