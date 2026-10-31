from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np
from numpy.typing import NDArray
import torch


def add_self_loops(hyperedges: Sequence, num_nodes: int) -> List[NDArray]:
    """Append a singleton hyperedge {v} for every node (same as prepare.add_self_loop)."""
    return [np.asarray(e, dtype=np.int64) for e in hyperedges] + [
        np.array([v], dtype=np.int64) for v in range(num_nodes)
    ]


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

