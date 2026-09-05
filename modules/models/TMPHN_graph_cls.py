from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
from typing import List, Dict
from torch_geometric.data import Data

from ..config import Params
from ..layers.tmessagepassing import TMessagePassing, Encoder


class TMPHN_Graph_Cls(nn.Module):
    def __init__(self, args: Params):
        super().__init__()
        self.num_layers = args.num_layers
        self.Mlst = args.Mlst
        self.hid_dim = args.hid_dim
        self.out_dim = args.num_classes
        self.device = torch.device(args.device)
        self.graph_readout_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim),
        )

    def build_encoder(self, X: torch.Tensor, neig_dict: Dict, args: Params):
        features_func = nn.Embedding(X.shape[0], X.shape[1])
        # NOTE: source of error here
        # features_func.weight = Parameter(X.clone(), requires_grad=True)
        features_func.weight.data.copy_(X)
        encoders: List[Encoder] = []
        for l in range(self.num_layers):
            if l == 0:
                agg = TMessagePassing(
                    features_func, neig_dict, self.Mlst[l], self.device
                )
                enc = Encoder(
                    features_func,
                    X.shape[1],
                    self.hid_dim,
                    args,
                    aggregator=agg,
                    base_model=None,
                )
            else:
                agg = TMessagePassing(
                    lambda n: encoders[l - 1](n), neig_dict, self.Mlst[l], self.device
                )
                enc = Encoder(
                    lambda n: encoders[l - 1](n),
                    encoders[l - 1].output_dim,
                    self.hid_dim,
                    args,
                    aggregator=agg,
                    base_model=encoders[l - 1],
                )
            encoders.append(enc)
        return encoders[-1]

    def forward(self, batch: List[Data], args: Params):
        graph_embeddings: List[torch.Tensor] = []
        for data in batch:
            X = data.x
            neig_dict = data.neig_dict
            enc: torch.nn.Module = self.build_encoder(X, neig_dict, args)
            node_emb: torch.Tensor = enc(torch.arange(X.size(0)))
            graph_emb = node_emb.mean(dim=0)
            graph_embeddings.append(graph_emb)
        graph_embeddings = torch.stack(graph_embeddings)  # [B, hid_dim]
        out = self.graph_readout_layer(graph_embeddings)  # [B, num_classes]
        return out
