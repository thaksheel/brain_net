import numpy as np
import math
from typing import Sequence

from .spectral import SpectralGraph, collate_spectral


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
