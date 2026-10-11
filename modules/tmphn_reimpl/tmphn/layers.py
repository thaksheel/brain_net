"""
T-MPHN building blocks.

Replaces: TMessagePassing.forward / aggregate_for_one_node  ->  tensor_message
          Encoder                                           ->  TMPHNLayer

Every node is processed at once with three GPU-friendly ops per layer:
gather (h[idx]), elementwise multiply (Hadamard product), scatter-add (index_add).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .data import Terms


def _weighted_products(h: torch.Tensor, idx: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """coef[t] * h[idx[t, 0]] * ... * h[idx[t, M-2]] for every term t  ->  (T, D)."""
    out = h[idx[:, 0]]
    for j in range(1, idx.shape[1]):
        out = out * h[idx[:, j]]  # Hadamard product = cross-node interaction (Eq. 21)
    return out * coef.to(out.dtype).unsqueeze(1)  # adjacency value / degree (Eqs. 18, 22)


def tensor_message(
    h: torch.Tensor,
    terms: Terms,
    chunk_size: Optional[int] = None,
    checkpoint: bool = False,
) -> torch.Tensor:
    """Neighborhood message m_N(v) for ALL nodes (Eqs. 21-24)  ->  (N, D).

    chunk_size : process terms in chunks to cap peak memory (None = all at once)
    checkpoint : recompute chunk products during backward instead of storing them
                 (much lower memory on large graphs, ~1 extra forward of compute)
    """
    if h.shape[0] != terms.num_nodes:
        raise ValueError(f"h has {h.shape[0]} rows but terms were compiled for {terms.num_nodes} nodes")
    out = h.new_zeros(h.shape)
    T = terms.num_terms
    step = chunk_size or max(T, 1)
    for start in range(0, T, step):
        sl = slice(start, start + step)
        if checkpoint and torch.is_grad_enabled():
            contrib = _checkpoint(_weighted_products, h, terms.idx[sl], terms.coef[sl], use_reentrant=False)
        else:
            contrib = _weighted_products(h, terms.idx[sl], terms.coef[sl])
        out = out.index_add(0, terms.target[sl], contrib)  # sum terms into their target nodes
    return out


def normalize(h: torch.Tensor, mode: str = "rms", eps: float = 1e-6) -> torch.Tensor:
    """Per-node scale control before the next layer multiplies vectors together.

    "rms"  : unit root-mean-square, so entries are O(1) and products of M-1 entries
             neither vanish nor explode (default)
    "l2"   : unit L2 norm, exactly the paper's Eq. 26 (entries ~ 1/sqrt(D), so the
             products shrink like D^{-(M-1)/2})
    "none" : no normalization
    """
    if mode == "rms":
        return h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    if mode == "l2":
        return F.normalize(h, p=2, dim=-1, eps=eps)
    if mode == "none":
        return h
    raise ValueError(f"unknown norm '{mode}'")


class TMPHNLayer(nn.Module):
    """One T-MPHN layer (paper Algorithm 2, Eqs. 24-26).

        m   = tensor_message(h)                         # Eq. 24
        z   = [h, m]  (concat)   or   h + m  (sum)      # COMBINE
        h'  = normalize(ReLU(Linear(dropout(z))))       # Eqs. 25-26

    Dropout is applied once, to the input of the linear layer. Messages are built
    from the un-dropped h, so dropout masks are never multiplied together.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        combine: str = "concat",
        dropout: float = 0.0,
        norm: str = "rms",
        chunk_size: Optional[int] = None,
        checkpoint: bool = False,
    ):
        super().__init__()
        if combine not in ("concat", "sum"):
            raise ValueError("combine must be 'concat' or 'sum'")
        self.combine = combine
        self.norm = norm
        self.chunk_size = chunk_size
        self.checkpoint = checkpoint
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * in_dim if combine == "concat" else in_dim, out_dim)

    def forward(self, h: torch.Tensor, terms: Terms) -> torch.Tensor:
        m = tensor_message(h, terms, self.chunk_size, self.checkpoint)
        z = torch.cat([h, m], dim=-1) if self.combine == "concat" else h + m
        z = F.relu(self.lin(self.dropout(z)))
        return normalize(z, self.norm)


def global_pool(h: torch.Tensor, batch: torch.Tensor, num_graphs: int, mode: str = "mean") -> torch.Tensor:
    """Node embeddings (N, D) -> graph embeddings (num_graphs, D)."""
    if mode in ("sum", "mean"):
        out = h.new_zeros(num_graphs, h.shape[1]).index_add(0, batch, h)
        if mode == "sum":
            return out
        count = torch.bincount(batch, minlength=num_graphs).clamp(min=1).to(h.dtype)
        return out / count.unsqueeze(1)
    if mode == "max":
        index = batch.unsqueeze(1).expand_as(h)
        init = h.new_full((num_graphs, h.shape[1]), float("-inf"))
        return init.scatter_reduce(0, index, h, reduce="amax")
    raise ValueError(f"unknown pool '{mode}'")
