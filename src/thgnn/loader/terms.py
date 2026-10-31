import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


@dataclass(repr=False)
class CompiledTerms:
    target: np.ndarray  # (T,)      int64: node that receives the term
    idx: np.ndarray  # (T, M-1)  int64: nodes whose vectors are multiplied
    coef: np.ndarray  # (T,)      float32: a_e * #orderings / degree(target)
    num_nodes: int
    M: int

    @property
    def num_terms(self) -> int:
        return int(self.target.shape[0])

    def __repr__(self) -> str:
        return f"CompiledTerms(M={self.M}, num_nodes={self.num_nodes}, num_terms={self.num_terms})"


@dataclass(repr=False)
class Terms:
    """Compiled message terms of one hypergraph for one order M, as tensors.

    message[target[t]] += coef[t] * h[idx[t, 0]] * ... * h[idx[t, M-2]]
    """

    target: torch.Tensor  # (T,)      long
    idx: torch.Tensor  # (T, M-1)  long
    coef: torch.Tensor  # (T,)      float32
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
        return (
            f"Terms(M={self.M}, num_nodes={self.num_nodes}, num_terms={self.num_terms})"
        )