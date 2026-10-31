import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalize


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
