from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Terms
from .utils import tensor_message, normalize


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
