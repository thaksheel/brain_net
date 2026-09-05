import torch.nn as nn
import torch
from typing import Dict
from torch_geometric.utils import subgraph

from ..config import Params
from .tmessagepassing import TMessagePassing, Encoder


class TMPHN_Graph_Cls(nn.Module):
    def __init__(self, args: Params):
        super().__init__()
        self.num_layers = args.num_layers
        self.Mlst = args.Mlst
        self.args = args
        self.hid_dim = args.hid_dim
        self.out_dim = args.num_classes
        self.device = torch.device(args.device)
        self.graph_readout_layer = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.hid_dim, self.out_dim),
        )
        # Shared encoder skeleton (no fixed X/neig_dict)
        self.encoders = nn.ModuleList()
        input_dim = args.num_features  # or args.input_dim if you set it
        # layer 0
        agg0 = TMessagePassing(
            features=None,  # will be set dynamically
            structure=None,  # will be set dynamically
            M=self.Mlst[0],
            device=self.device,
        )
        enc0 = Encoder(
            features=None,
            input_dim=input_dim,
            output_dim=self.hid_dim,
            args=self.args,
            aggregator=agg0,
            base_model=None,
        )
        self.encoders.append(enc0)
        # subsequent layers
        for l in range(1, self.num_layers):
            agg = TMessagePassing(
                features=None,
                structure=None,
                M=self.Mlst[l],
                device=self.device,
            )
            enc = Encoder(
                features=None,
                input_dim=self.hid_dim,
                output_dim=self.hid_dim,
                args=self.args,
                aggregator=agg,
                base_model=self.encoders[l - 1],
            )
            self.encoders.append(enc)

    def encode_graph(self, X: torch.Tensor, neig_dict: Dict) -> torch.Tensor:
        nodes = torch.arange(X.size(0), device=self.device)
        # layer 0
        h = self.encoders[0].forward_with_features(X, neig_dict, nodes)
        h = nn.functional.dropout(h, p=self.args.dropout, training=self.training)
        # subsequent layers
        for l in range(1, self.num_layers):
            h = self.encoders[l].forward_with_features(h, neig_dict, nodes)
            h = nn.functional.dropout(h, p=self.args.dropout, training=self.training)
        return h

    def build_neig_dict_from_edge_index(self, edge_index, num_nodes):
        neig_dict = {i: [] for i in range(num_nodes)}
        E = edge_index.size(1)
        for e in range(E):
            u = edge_index[0, e].item()
            v = edge_index[1, e].item()
            neig_dict[u].append([u, v])
            neig_dict[v].append([u, v])
        return neig_dict

    def forward(self, batch):
        X = batch.x.to(self.device)
        edge_index = batch.edge_index
        batch_index = batch.batch
        graph_embeddings = []
        for graph_id in batch_index.unique():
            node_mask = batch_index == graph_id
            nodes = torch.where(node_mask)[0]
            X_g = X[nodes]
            edge_index_g, _ = subgraph(
                subset=nodes,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=batch.x.size(0),
            )
            neig_dict = self.build_neig_dict_from_edge_index(edge_index_g, X_g.size(0))
            node_emb = self.encode_graph(X_g, neig_dict)
            graph_emb = node_emb.mean(dim=0)
            graph_embeddings.append(graph_emb)
        graph_embeddings = torch.stack(graph_embeddings)
        return self.graph_readout_layer(graph_embeddings)

