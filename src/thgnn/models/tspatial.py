import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TSpatialLayer, normalize
from ..loader import (
    collate_spectral,
    SpectralGraph,
    SpectralBatch,
    adjacency_tensor,
)
from ...config import Params


class TSpatialHGNN(nn.Module):
    """
        x --Linear--> normalize --> h0 --TSpatialLayer x L--> hL --(pool)--> Linear --> log_softmax

    forward(SpectralGraph) -> (N, C) for task="node"; forward(SpectralBatch) -> (B, C) for task="graph".
    Same constructor and calling convention as TSpectralHGNN (no max_nodes needed).
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        combine="sum",
        dropout=0.3,
        norm="rms",
        task="node",
        pool="mean",
        project_input=True,
    ):
        super().__init__()
        if task not in ("node", "graph"):
            raise ValueError("task must be 'node' or 'graph'")
        self.task, self.pool, self.norm = task, pool, norm
        self.input_proj = nn.Linear(in_dim, hid_dim) if project_input else None
        dims = [hid_dim if project_input else in_dim] + [hid_dim] * num_layers
        if combine == "sum" and dims[0] != dims[1]:
            raise ValueError(
                "combine='sum' needs project_input=True (or in_dim == hid_dim)"
            )
        self.layers = nn.ModuleList(
            [
                TSpatialLayer(dims[l], dims[l + 1], combine, dropout, norm)
                for l in range(num_layers)
            ]
        )
        self.head = nn.Linear(hid_dim, num_classes)

    @classmethod
    def from_params(cls, params: Params, **overrides) -> "TSpatialHGNN":
        kwargs = dict(
            in_dim=params.num_features,
            hid_dim=params.hid_dim,
            num_classes=params.num_classes,
            num_layers=params.num_layers,
            combine=params.combine,
            dropout=params.dropout,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def reset_parameters(self):
        if self.input_proj is not None:
            self.input_proj.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.head.reset_parameters()

    def embed(self, batch: SpectralBatch) -> torch.Tensor:
        x = batch.x
        N = x.shape[1]
        A = adjacency_tensor(batch, N, x.dtype)  # fixed structure, shared by all layers
        h = x if self.input_proj is None else normalize(self.input_proj(x), self.norm)
        for layer in self.layers:
            h = layer(h, A)
        return h  # (B, N, hid)

    def forward(self, data) -> torch.Tensor:
        batch = collate_spectral([data]) if isinstance(data, SpectralGraph) else data
        h = self.embed(batch)
        if self.task == "node":
            out = self.head(h)
            return F.log_softmax(
                out[0] if isinstance(data, SpectralGraph) else out, dim=-1
            )
        if self.pool == "mean":
            g = h.mean(1)
        elif self.pool == "sum":
            g = h.sum(1)
        elif self.pool == "max":
            g = h.amax(1)
        else:
            raise ValueError(f"unknown pool '{self.pool}'")
        return F.log_softmax(self.head(g), dim=-1)
