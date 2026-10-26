import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import normalize
from .tspectral import SpectralBatch, SpectralGraph, collate_spectral


def adjacency_tensor(batch: SpectralBatch, N: int, dtype=torch.float32) -> torch.Tensor:
    """Dense degree-normalized adjacency tensor a[b, i, j, k] from the compiled terms -> (B, N, N, N).

    Each term (target t, partners (p, q), coef) is a[t, p, q] = a[t, q, p] = coef / 2.
    """
    B = batch.num_graphs
    t, p, q = batch.target, batch.idx[:, 0], batch.idx[:, 1]
    w = (batch.coef / 2).to(dtype)
    base = (batch.graph * N + t) * N
    A = torch.zeros(B * N * N * N, dtype=dtype, device=batch.x.device)
    A.index_add_(0, (base + p) * N + q, w)
    A.index_add_(0, (base + q) * N + p, w)
    return A.view(B, N, N, N)


class TSpatialLayer(nn.Module):
    """One T-spatial layer:

    y  = sum_{j,k} a[:, j, k] h_j ⊙ h_k                     Eq. 13 (first slice of the t-product)
    z  = h + y ("sum"), [h, y] ("concat"), or y ("none", the original layer)
    h' = normalize(ReLU(Linear(dropout(z))))
    """

    def __init__(self, in_dim, out_dim, combine="sum", dropout=0.0, norm="rms"):
        super().__init__()
        if combine not in ("sum", "concat", "none"):
            raise ValueError("combine must be 'sum', 'concat' or 'none'")
        self.combine, self.norm = combine, norm
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * in_dim if combine == "concat" else in_dim, out_dim)

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def message(h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """h (B, N, D), A (B, N, N, N) -> y (B, N, D)."""
        Ah = torch.einsum("bijk,bjd->bikd", A, h)  # contract j first: (B, N, N, D)
        return torch.einsum("bikd,bkd->bid", Ah, h)  # then the second partner k

    def forward(self, h, A):
        y = self.message(h, A)
        if self.combine == "concat":
            z = torch.cat([h, y], dim=-1)
        elif self.combine == "sum":
            z = h + y
        else:
            z = y
        return normalize(F.relu(self.lin(self.dropout(z))), self.norm)


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
    def from_params(cls, params, **overrides) -> "TSpatialHGNN":
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
