import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Dict, List

from . import TMessagePassing, Encoder
from ..config import Params


class TMPHN(nn.Module):
    """
    The mean aggregator for a hypergraph
    """

    def __init__(self, X: torch.Tensor, neig_dict: Dict, args: Params):
        """
        features: a function mapping LongTensor of node ids to FloatTensor of feature values
        structure: a dictionary store the neighbors for all target nodes,
                i.e., key is target node, value is hyperedges contain the target node
        M: the maximum cardinality of the hypergraph
        """
        super(TMPHN, self).__init__()

        self.num_layers = args.num_layers
        self.Mlst = args.Mlst
        assert self.num_layers == len(
            self.Mlst
        ), "The number of layers should be equal to the length of Mlst"
        self.input_dim = X.shape[1]
        self.hid_dim = args.hid_dim
        self.out_dim = args.num_classes
        self.device = torch.device(args.device)

        # initialize TMPHN layers
        hidden_dim = args.hid_dim
        features_func = nn.Embedding(X.shape[0], X.shape[1])
        features_func.weight = nn.Parameter(torch.FloatTensor(X), requires_grad=False)
        encoders: List[Encoder] = []
        for l in range(self.num_layers):
            if l == 0:  # for the first layer
                agg = TMessagePassing(
                    features_func, neig_dict, self.Mlst[l], self.device
                )
                enc = Encoder(
                    features_func,
                    X.shape[1],
                    hidden_dim,
                    args,
                    aggregator=agg,
                    base_model=None,
                )
            else:  # for the subsequent layers
                prev = encoders[l - 1]
                agg = TMessagePassing(prev, neig_dict, self.Mlst[l], self.device)
                enc = Encoder(
                    prev,
                    prev.output_dim,
                    hidden_dim,
                    args,
                    aggregator=agg,
                    base_model=prev,
                )
            encoders.append(enc)  # add the created encoder to the list
        self.encoder: Encoder = encoders[-1]

        # MLP layers for readout
        self.W = nn.Parameter(torch.FloatTensor(self.encoder.output_dim, self.out_dim))
        self.b = nn.Parameter(torch.FloatTensor(self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.b.data.uniform_(-std, std)

    def forward(self, nodes):
        X = self.encoder.forward(nodes)
        W, b = self.W.to(self.device), self.b.to(self.device)
        y_pred = torch.matmul(X, W) + b
        return F.log_softmax(y_pred, dim=1)
