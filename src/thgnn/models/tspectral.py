import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import TSpectralLayer, normalize
from ..loader import (
    collate_spectral,
    SpectralGraph,
    cosine_basis,
    adjacency_spectrum,
    SpectralBatch,
)
from ...config import Params


class TSpectralHGNN(nn.Module):
    """
        x --Linear--> normalize --> h0 --TSpectralLayer x L--> hL --(mean/sum/max pool)--> Linear --> log_softmax

    forward(batch) with a SpectralGraph (node task: (N, C) output) or a SpectralBatch
    (graph task: (B, C) output), matching how GraphTrainer calls TMPHN.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=1,
        max_nodes: int = 1000,
        combine="concat",
        dropout=0.3,
        norm="rms",
        readout="learned",
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
        self.layers = nn.ModuleList(
            [
                TSpectralLayer(
                    dims[l], dims[l + 1], max_nodes, combine, dropout, norm, readout
                )
                for l in range(num_layers)
            ]
        )
        self.head = nn.Linear(hid_dim, num_classes)

    @classmethod
    def from_params(
        cls, params: Params, max_nodes: int, **overrides
    ) -> "TSpectralHGNN":
        kwargs = dict(
            in_dim=params.num_features,
            hid_dim=params.hid_dim,
            num_classes=params.num_classes,
            num_layers=params.num_layers,
            max_nodes=max_nodes,
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
        B, N, _ = x.shape
        if N > self.layers[0].max_nodes:
            raise ValueError(
                f"graph has {N} nodes but the model was built with max_nodes={self.layers[0].max_nodes}"
            )
        C = cosine_basis(N, device=x.device, dtype=x.dtype)
        A_hat = adjacency_spectrum(batch, N, C)  # fixed structure, shared by all layers
        h = x if self.input_proj is None else normalize(self.input_proj(x), self.norm)
        for layer in self.layers:
            h = layer(h, A_hat, C)
        return h  # (B, N, hid)

    def forward(self, data) -> torch.Tensor:
        batch = collate_spectral([data]) if isinstance(data, SpectralGraph) else data
        h = self.embed(batch)
        if self.task == "node":
            out = self.head(h)  # (B, N, C)
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
        return F.log_softmax(self.head(g), dim=-1)  # (B, C)
