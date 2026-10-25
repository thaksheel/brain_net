"""
T-spectral HyperGNN (paper Sec. III-B, Eq. 6-10, Appendix B).

Order M = 3 (the setting the paper uses for T-spectral). The adjacency tensor is built from
the same compiled terms as T-MPHN (so oversized hyperedges are down-sampled the same way and
every node's weights sum to 1).

Per layer, for each subject (b), node i, frequency f, feature d:
    A_hat[b, i, j, f] = sum_k A[b, i, j, k] * c_f(k)        (fixed, built once per batch)
    h_hat[b, f, d]    = sum_k h[b, k, d] * c_f(k)           c_f(k) = 2 cos(2 pi f (k+1) / Ns)
    Y_hat[b, i, f, d] = h_hat[b, f, d] * sum_j A_hat[b, i, j, f] h[b, j, d]
    m[b, i, d]        = sum_f w_f Y_hat[b, i, f, d]         (learned spectral filter)

Cost: memory O(B N^2 N) for A_hat and time O(B N^3 D) per layer.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import normalize
from .structure import add_self_loops, compile_terms


@dataclass(repr=False)
class SpectralGraph:
    x: torch.Tensor  # (N, F)
    target: torch.Tensor  # (T,)   compiled terms (M = 3), as in T-MPHN
    idx: torch.Tensor  # (T, 2)
    coef: torch.Tensor  # (T,)
    y: Optional[torch.Tensor] = None  # (N,) node labels or (1,) graph label

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device, non_blocking: bool = False) -> "SpectralGraph":
        mv = lambda t: None if t is None else t.to(device, non_blocking=non_blocking)
        return SpectralGraph(
            mv(self.x), mv(self.target), mv(self.idx), mv(self.coef), mv(self.y)
        )

    def __repr__(self):
        return f"SpectralGraph(num_nodes={self.num_nodes}, features={self.x.shape[1]}, terms={self.target.numel()})"


def build_spectral_graph(
    x,
    hyperedges: Sequence,
    y=None,
    self_loops: bool = False,
    strategy: str = "random",
    scores=None,
    temperature: float = 1.0,
    seed: Optional[int] = 0,
) -> SpectralGraph:
    """Compile one hypergraph for TSpectralHGNN (same arguments as build_hypergraph, M fixed to 3)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    n = int(x.shape[0])
    if self_loops:
        hyperedges = add_self_loops(hyperedges, n)
    ct = compile_terms(
        hyperedges, n, 3, strategy, scores, temperature, np.random.default_rng(seed)
    )
    if y is not None:
        y = torch.as_tensor(np.asarray(y), dtype=torch.long).reshape(-1)
    return SpectralGraph(
        x,
        torch.from_numpy(ct.target),
        torch.from_numpy(ct.idx),
        torch.from_numpy(ct.coef),
        y,
    )


@dataclass(repr=False)
class SpectralBatch:
    """B hypergraphs with the same number of nodes, stacked (not merged)."""

    x: torch.Tensor  # (B, N, F)
    target: torch.Tensor  # (T_total,) node ids within each graph
    idx: torch.Tensor  # (T_total, 2)
    coef: torch.Tensor  # (T_total,)
    graph: torch.Tensor  # (T_total,) which graph each term belongs to
    y: Optional[torch.Tensor]  # (B,) graph labels
    num_graphs: int

    def to(self, device, non_blocking: bool = False) -> "SpectralBatch":
        mv = lambda t: None if t is None else t.to(device, non_blocking=non_blocking)
        return SpectralBatch(
            mv(self.x),
            mv(self.target),
            mv(self.idx),
            mv(self.coef),
            mv(self.graph),
            mv(self.y),
            self.num_graphs,
        )


def collate_spectral(graphs: Sequence[SpectralGraph]) -> SpectralBatch:
    sizes = {g.num_nodes for g in graphs}
    if len(sizes) != 1:
        raise ValueError(
            f"T-spectral batches need graphs with equal node counts, got {sorted(sizes)}; "
            "use batch_size=1 or a fixed atlas"
        )
    ys = [g.y for g in graphs]
    return SpectralBatch(
        x=torch.stack([g.x for g in graphs]),
        target=torch.cat([g.target for g in graphs]),
        idx=torch.cat([g.idx for g in graphs]),
        coef=torch.cat([g.coef for g in graphs]),
        graph=torch.cat([torch.full_like(g.target, i) for i, g in enumerate(graphs)]),
        y=torch.cat(ys) if all(t is not None for t in ys) else None,
        num_graphs=len(graphs),
    )


class SpectralGraphLoader:
    """Same interface as HyperGraphLoader: iterate SpectralBatch objects."""

    def __init__(
        self,
        graphs: Sequence[SpectralGraph],
        batch_size: int = 16,
        shuffle: bool = True,
        seed=None,
    ):
        self.graphs, self.batch_size, self.shuffle = list(graphs), batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return math.ceil(len(self.graphs) / self.batch_size)

    def __iter__(self):
        n = len(self.graphs)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for s in range(0, n, self.batch_size):
            yield collate_spectral(
                [self.graphs[i] for i in order[s : s + self.batch_size]]
            )


# =============================================================================
# Spectral operators
# =============================================================================
def cosine_basis(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """c_f(k) = 2 cos(2 pi f (k+1) / (2N+1)) for nodes-as-slices k = 0..N-1 and frequencies f = 0..N -> (N, N+1).

    This is the (real) DFT of the symmetrized sequence [0, s_1..s_N, s_N..s_1]; using it
    replaces fft/ifft and keeps everything real.
    """
    Ns = 2 * N + 1
    k = torch.arange(1, N + 1, device=device, dtype=torch.float64)
    f = torch.arange(N + 1, device=device, dtype=torch.float64)
    return (2 * torch.cos(2 * math.pi * torch.outer(k, f) / Ns)).to(dtype)


def adjacency_spectrum(batch: SpectralBatch, N: int, C: torch.Tensor) -> torch.Tensor:
    """A_hat[b, i, j, f] = sum_k A[b, i, j, k] c_f(k), built directly from the terms -> (B, N, N, N+1).

    Each term (target t, partners (a, b), coef) is the adjacency entries a[t,a,b] = a[t,b,a] = coef/2.
    """
    B, Nf = batch.num_graphs, C.shape[1]
    t, a, b = batch.target, batch.idx[:, 0], batch.idx[:, 1]
    w = (batch.coef / 2).to(C.dtype).unsqueeze(1)
    out = C.new_zeros(B * N * N, Nf)
    base = (batch.graph * N + t) * N
    out.index_add_(0, base + a, w * C[b])  # entry a[t, a, b]: row j = a, slice k = b
    out.index_add_(0, base + b, w * C[a])  # entry a[t, b, a]: row j = b, slice k = a
    return out.view(B, N, N, Nf)


class TSpectralLayer(nn.Module):
    """One T-spectral layer:

        m  = sum_f w_f * h_hat[f] ⊙ (A_hat[:, :, f] @ h)       spectral convolution + learned filter
        h' = normalize(ReLU(Linear(dropout([h, m]))))          same combine as T-MPHN

    readout:
        "learned" : w_f learnable, initialized to the first slice (= T-spatial / T-MPHN message)
        "first"   : fixed first slice (exactly T-spatial)
        "sum"     : the original code's sum over slices (frequency 0 only), divided by Ns for scale
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        max_nodes: int,
        combine="concat",
        dropout=0.0,
        norm="rms",
        readout="learned",
    ):
        super().__init__()
        if readout not in ("learned", "first", "sum"):
            raise ValueError("readout must be 'learned', 'first' or 'sum'")
        self.combine, self.norm, self.readout = combine, norm, readout
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * in_dim if combine == "concat" else in_dim, out_dim)
        self.max_nodes = max_nodes
        self.filter = (
            nn.Parameter(torch.empty(max_nodes + 1)) if readout == "learned" else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.filter is not None:
            with torch.no_grad():
                self.filter.copy_(
                    self._first_slice_weights(self.max_nodes, self.filter.device)
                )

    @staticmethod
    def _first_slice_weights(N: int, device=None) -> torch.Tensor:
        Ns = 2 * N + 1
        w = torch.full((N + 1,), 2.0 / Ns, device=device)
        w[0] = 1.0 / Ns
        return w / 2  # slice 0 of the symmetrized product counts every pair twice

    def message(
        self, h: torch.Tensor, A_hat: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """h (B, N, D), A_hat (B, N, N, Nf), C (N, Nf) -> m (B, N, D)."""
        N = h.shape[1]
        h_hat = torch.einsum("bkd,kf->bfd", h, C)  # spectrum of the symmetrized signal
        AH = torch.einsum("bijf,bjd->bfid", A_hat, h)  # per-frequency A_f @ h
        if self.readout == "sum":
            return AH[:, 0] * h_hat[:, 0].unsqueeze(1) / (2 * N + 1)
        w = (
            self.filter[: N + 1]
            if self.readout == "learned"
            else self._first_slice_weights(N, h.device)
        )
        return torch.einsum("bfid,bfd,f->bid", AH, h_hat, w.to(h.dtype))

    def forward(self, h, A_hat, C):
        m = self.message(h, A_hat, C)
        z = torch.cat([h, m], dim=-1) if self.combine == "concat" else h + m
        return normalize(F.relu(self.lin(self.dropout(z))), self.norm)


# =============================================================================
# Model
# =============================================================================
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
    def from_params(cls, params, max_nodes: int, **overrides) -> "TSpectralHGNN":
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
