"""
Torch-side containers: what the model consumes.

    build_hypergraph(x, hyperedges, Mlst)  ->  HyperGraph   (one hypergraph, terms for every layer)
    collate([HyperGraph, ...])             ->  HyperBatch   (disjoint union, for graph classification)
    HyperGraphLoader(graphs, batch_size)   ->  iterates HyperBatch objects

Replaces: the neig_dict plumbing in prepare.py, Encoder.forward_with_features,
and config.HypergraphBatchLoader.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .structure import CompiledTerms, add_self_loops, compile_terms


@dataclass(repr=False)
class Terms:
    """Compiled message terms of one hypergraph for one order M, as tensors.

    message[target[t]] += coef[t] * h[idx[t, 0]] * ... * h[idx[t, M-2]]
    """

    target: torch.Tensor  # (T,)      long
    idx: torch.Tensor     # (T, M-1)  long
    coef: torch.Tensor    # (T,)      float32
    num_nodes: int
    M: int

    @classmethod
    def from_compiled(cls, ct: CompiledTerms) -> "Terms":
        return cls(
            torch.from_numpy(ct.target),
            torch.from_numpy(ct.idx),
            torch.from_numpy(ct.coef),
            ct.num_nodes,
            ct.M,
        )

    def to(self, device, non_blocking: bool = False) -> "Terms":
        return Terms(
            self.target.to(device, non_blocking=non_blocking),
            self.idx.to(device, non_blocking=non_blocking),
            self.coef.to(device, non_blocking=non_blocking),
            self.num_nodes,
            self.M,
        )

    @staticmethod
    def batch(parts: Sequence["Terms"]) -> "Terms":
        """Disjoint union: shift each graph's node ids by the number of nodes before it."""
        M = parts[0].M
        if any(p.M != M for p in parts):
            raise ValueError("cannot batch Terms compiled with different M")
        offsets = np.cumsum([0] + [p.num_nodes for p in parts[:-1]])
        return Terms(
            target=torch.cat([p.target + int(o) for p, o in zip(parts, offsets)]),
            idx=torch.cat([p.idx + int(o) for p, o in zip(parts, offsets)]),
            coef=torch.cat([p.coef for p in parts]),
            num_nodes=sum(p.num_nodes for p in parts),
            M=M,
        )

    @property
    def num_terms(self) -> int:
        return int(self.target.numel())

    def __repr__(self) -> str:
        return f"Terms(M={self.M}, num_nodes={self.num_nodes}, num_terms={self.num_terms})"


def _move_terms(terms: Sequence[Terms], device, non_blocking: bool) -> List[Terms]:
    """Move per-layer terms; layers that share one structure (same M) stay shared."""
    moved = {}
    for t in terms:
        if id(t) not in moved:
            moved[id(t)] = t.to(device, non_blocking)
    return [moved[id(t)] for t in terms]


@dataclass(repr=False)
class HyperGraph:
    """One hypergraph ready for the model."""

    x: torch.Tensor                   # (N, F) node features
    terms: List[Terms]                # terms[l] feeds layer l (compiled with Mlst[l])
    y: Optional[torch.Tensor] = None  # (N,) node labels, or (1,) graph label

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to(self, device, non_blocking: bool = False) -> "HyperGraph":
        return HyperGraph(
            self.x.to(device, non_blocking=non_blocking),
            _move_terms(self.terms, device, non_blocking),
            None if self.y is None else self.y.to(device, non_blocking=non_blocking),
        )

    def __repr__(self) -> str:
        return f"HyperGraph(num_nodes={self.num_nodes}, features={self.x.shape[1]}, terms={self.terms})"


def build_hypergraph(
    x,
    hyperedges: Sequence,
    Mlst: Sequence[int],
    y=None,
    self_loops: bool = False,
    strategy: str = "random",
    scores=None,
    temperature: float = 1.0,
    seed: Optional[int] = 0,
) -> HyperGraph:
    """Compile one hypergraph for a model with layer orders `Mlst`.

    Each distinct M is compiled once; hyperedges larger than a layer's M are
    down-sampled for that layer with `strategy` ("random" | "topk" | "weighted",
    the last two use `scores[target, member]`, e.g. |FC| for brain data).
    Re-call with a new `seed` (e.g. every few epochs) to resample neighbors.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    n = int(x.shape[0])
    if self_loops:
        hyperedges = add_self_loops(hyperedges, n)
    rng = np.random.default_rng(seed)
    compiled = {}
    for M in Mlst:
        if M not in compiled:
            ct = compile_terms(hyperedges, n, M, strategy, scores, temperature, rng)
            compiled[M] = Terms.from_compiled(ct)
    if y is not None:
        y = torch.as_tensor(np.asarray(y), dtype=torch.long).reshape(-1)
    return HyperGraph(x, [compiled[M] for M in Mlst], y)


@dataclass(repr=False)
class HyperBatch:
    """Several hypergraphs merged into one disconnected hypergraph."""

    x: torch.Tensor               # (N_total, F)
    terms: List[Terms]            # per layer, node ids already shifted
    y: Optional[torch.Tensor]     # (num_graphs,) for graph classification
    batch: torch.Tensor           # (N_total,) graph id of every node
    num_graphs: int

    def to(self, device, non_blocking: bool = False) -> "HyperBatch":
        return HyperBatch(
            self.x.to(device, non_blocking=non_blocking),
            _move_terms(self.terms, device, non_blocking),
            None if self.y is None else self.y.to(device, non_blocking=non_blocking),
            self.batch.to(device, non_blocking=non_blocking),
            self.num_graphs,
        )


def collate(graphs: Sequence[HyperGraph]) -> HyperBatch:
    x = torch.cat([g.x for g in graphs])
    by_M = {}  # layers with the same M share one structure (see build_hypergraph)
    for l, t in enumerate(graphs[0].terms):
        if t.M not in by_M:
            by_M[t.M] = Terms.batch([g.terms[l] for g in graphs])
    terms = [by_M[t.M] for t in graphs[0].terms]
    y = torch.cat([g.y for g in graphs]) if all(g.y is not None for g in graphs) else None
    batch = torch.cat(
        [torch.full((g.num_nodes,), i, dtype=torch.long, device=g.x.device) for i, g in enumerate(graphs)]
    )
    return HyperBatch(x, terms, y, batch, len(graphs))


class HyperGraphLoader:
    """Mini-batches of whole hypergraphs (e.g. one hypergraph per subject)."""

    def __init__(self, graphs: Sequence[HyperGraph], batch_size: int = 16, shuffle: bool = True, seed=None):
        self.graphs = list(graphs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.graphs) / self.batch_size)

    def __iter__(self):
        n = len(self.graphs)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            yield collate([self.graphs[i] for i in order[start : start + self.batch_size]])
