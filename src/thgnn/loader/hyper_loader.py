from typing import List, Optional, Sequence
import numpy as np
import torch
import math

from . import HyperGraph, collate


class HyperGraphLoader:
    """Mini-batches of whole hypergraphs (e.g. one hypergraph per subject)."""

    def __init__(
        self,
        graphs: Sequence[HyperGraph],
        batch_size: int = 16,
        shuffle: bool = True,
        seed=None,
    ):
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
            yield collate(
                [self.graphs[i] for i in order[start : start + self.batch_size]]
            )
