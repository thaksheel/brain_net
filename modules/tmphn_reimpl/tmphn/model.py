"""
The T-MPHN model (replaces TMPHN.py).

    x --Linear--> normalize --> h0                       # x(0) = MLP(x)     (Algorithm 2, line 1)
    h0 --TMPHNLayer(M = Mlst[0])--> h1                   # each layer runs ONCE for ALL nodes
    ...
    h(L-1) --TMPHNLayer(M = Mlst[L-1])--> hL             # node embeddings
    hL --(global pool, graph task only)--> Linear --> log_softmax

The layers are a plain nn.ModuleList called in a for-loop: no recursion,
no layer holds a reference to another layer, and every parameter appears
exactly once in state_dict().
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Terms
from .layers import TMPHNLayer, global_pool, normalize


class TMPHN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_classes: int,
        Mlst: Sequence[int],
        combine: str = "concat",
        dropout: float = 0.3,
        norm: str = "rms",
        task: str = "node",
        pool: str = "mean",
        chunk_size: Optional[int] = None,
        checkpoint: bool = False,
        project_input: bool = True,
    ):
        """
        in_dim      : input feature dimension
        hid_dim     : embedding dimension of every layer
        num_classes : output classes
        Mlst        : hypergraph order used by each layer (len(Mlst) = number of layers)
        combine     : "concat" (paper) or "sum" of self and message vectors
        dropout     : dropout on each layer's input to its linear transform
        norm        : "rms" (default), "l2" (paper Eq. 26) or "none"
        task        : "node" -> per-node log-probs;  "graph" -> per-graph log-probs
        pool        : graph readout ("mean" | "sum" | "max"), used when task="graph"
        chunk_size, checkpoint : memory controls for large graphs (see layers.tensor_message)
        project_input : apply the paper's input Linear (x0 = MLP(x)); False feeds raw x to layer 0
                        like the original code (useful for ablations)
        """
        super().__init__()
        if task not in ("node", "graph"):
            raise ValueError("task must be 'node' or 'graph'")
        self.Mlst: List[int] = [int(M) for M in Mlst]
        self.task = task
        self.pool = pool
        self.norm = norm
        self.input_proj = nn.Linear(in_dim, hid_dim) if project_input else None
        dims = [hid_dim if project_input else in_dim] + [hid_dim] * len(self.Mlst)
        self.layers = nn.ModuleList(
            [
                TMPHNLayer(dims[l], dims[l + 1], combine, dropout, norm, chunk_size, checkpoint)
                for l in range(len(self.Mlst))
            ]
        )
        self.head = nn.Linear(hid_dim, num_classes)

    @classmethod
    def from_params(cls, params, **overrides) -> "TMPHN":
        """Build from your config.Params (uses num_features, hid_dim, num_classes, Mlst, combine, dropout)."""
        if params.num_layers != len(params.Mlst):
            raise ValueError("Params.num_layers must equal len(Params.Mlst)")
        kwargs = dict(
            in_dim=params.num_features,
            hid_dim=params.hid_dim,
            num_classes=params.num_classes,
            Mlst=params.Mlst,
            combine=params.combine,
            dropout=params.dropout,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def embed(self, x: torch.Tensor, terms: Sequence[Terms]) -> torch.Tensor:
        """Node embeddings (N, hid_dim) after the last layer."""
        if len(terms) != len(self.layers):
            raise ValueError(f"model has {len(self.layers)} layers but got terms for {len(terms)}")
        h = x if self.input_proj is None else normalize(self.input_proj(x), self.norm)
        for layer, t, M in zip(self.layers, terms, self.Mlst):
            if t.M != M:
                raise ValueError(f"layer expects terms compiled with M={M}, got M={t.M}")
            h = layer(h, t)
        return h

    def forward(
        self,
        x: torch.Tensor,
        terms: Sequence[Terms],
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> torch.Tensor:
        """Log-probabilities: (N, num_classes) for task="node", (num_graphs, num_classes) for task="graph"."""
        h = self.embed(x, terms)
        if self.task == "graph":
            if batch is None:  # a single graph
                batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
                num_graphs = 1
            h = global_pool(h, batch, num_graphs, self.pool)
        return F.log_softmax(self.head(h), dim=-1)
