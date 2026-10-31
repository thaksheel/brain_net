import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalize


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
