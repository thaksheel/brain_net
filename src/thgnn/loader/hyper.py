import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from . import Terms 


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

    x: torch.Tensor  # (N, F) node features
    terms: List[Terms]  # terms[l] feeds layer l (compiled with Mlst[l])
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


@dataclass(repr=False)
class HyperBatch:
    """Several hypergraphs merged into one disconnected hypergraph."""

    x: torch.Tensor  # (N_total, F)
    terms: List[Terms]  # per layer, node ids already shifted
    y: Optional[torch.Tensor]  # (num_graphs,) for graph classification
    batch: torch.Tensor  # (N_total,) graph id of every node
    num_graphs: int

    def to(self, device, non_blocking: bool = False) -> "HyperBatch":
        return HyperBatch(
            self.x.to(device, non_blocking=non_blocking),
            _move_terms(self.terms, device, non_blocking),
            None if self.y is None else self.y.to(device, non_blocking=non_blocking),
            self.batch.to(device, non_blocking=non_blocking),
            self.num_graphs,
        )


