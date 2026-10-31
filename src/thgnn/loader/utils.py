from typing import List, Optional, Sequence
import numpy as np
import math 
import torch
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb, factorial, prod
from typing import List, Optional, Sequence
from numpy.typing import NDArray

from . import Terms, HyperBatch, HyperGraph, SpectralBatch, SpectralGraph


def collate(graphs: Sequence[HyperGraph]) -> HyperBatch:
    x = torch.cat([g.x for g in graphs])
    by_M = {}  # layers with the same M share one structure (see build_hypergraph)
    for l, t in enumerate(graphs[0].terms):
        if t.M not in by_M:
            by_M[t.M] = Terms.batch([g.terms[l] for g in graphs])
    terms = [by_M[t.M] for t in graphs[0].terms]
    y = (
        torch.cat([g.y for g in graphs])
        if all(g.y is not None for g in graphs)
        else None
    )
    batch = torch.cat(
        [
            torch.full((g.num_nodes,), i, dtype=torch.long, device=g.x.device)
            for i, g in enumerate(graphs)
        ]
    )
    return HyperBatch(x, terms, y, batch, len(graphs))


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

def hyperedges_from_incidence(H) -> List[NDArray]:
    """Incidence matrix H (N x E, dense or scipy.sparse) -> one node array per column."""
    if hasattr(H, "tocsc"):  # scipy.sparse
        H = H.tocsc(copy=True)
        H.eliminate_zeros()
        return [
            np.sort(H.indices[H.indptr[j] : H.indptr[j + 1]]).astype(np.int64)
            for j in range(H.shape[1])
        ]
    H = np.asarray(H)
    return [np.flatnonzero(H[:, j]).astype(np.int64) for j in range(H.shape[1])]


def hyperedges_from_adjacency(A) -> List[NDArray]:
    """One hyperedge per node: the node together with its graph neighbors."""
    A = (np.asarray(A) != 0) | np.eye(len(A), dtype=bool)
    rows, cols = np.nonzero(A)  # row-major, so each row's columns come out sorted
    return np.split(cols.astype(np.int64), np.flatnonzero(np.diff(rows)) + 1)


def knn_hyperedges(similarity, k: int) -> List[NDArray]:
    """One hyperedge per node: the node plus its k most similar other nodes.

    This is the construction used for the paper's 3D-object datasets (K = 5).
    For brain data pass |FC| so each ROI is grouped with its k most strongly
    connected ROIs.
    """
    S = np.array(similarity, dtype=np.float64)  # copy
    np.fill_diagonal(S, -np.inf)
    nearest = np.argsort(-S, axis=1)[:, :k]
    return [
        np.concatenate(([i], nearest[i])).astype(np.int64) for i in range(S.shape[0])
    ]


def induced_hyperedges(hyperedges: Sequence, nodes, num_nodes: int) -> List[NDArray]:
    """Keep only `nodes` inside every hyperedge (node ids are NOT relabeled).

    Use it for inductive training: compile the training structure from the nodes
    visible during training, and the evaluation structure from all nodes.
    Hyperedges that become empty are dropped.
    """
    keep = np.zeros(num_nodes, dtype=bool)
    keep[np.asarray(nodes, dtype=np.int64)] = True
    kept = [np.asarray(e, dtype=np.int64) for e in hyperedges]
    kept = [e[keep[e]] for e in kept]
    return [e for e in kept if e.size > 0]


# =============================================================================
# 2. Paper coefficients
# =============================================================================
def adjacency_value(c: int, M: int) -> float:
    """a_e = |e| / alpha  (Theorem 4.1, Eq. 18).

    alpha counts the ways of filling M index slots with the c nodes of the edge
    so that every node appears at least once.
    """
    alpha = sum((-1) ** i * comb(c, i) * (c - i) ** M for i in range(c + 1))
    return c / alpha


@lru_cache(maxsize=None)
def expansion_patterns(c: int, M: int, p: int):
    """All products contributed by an edge of size c <= M to its member at position p.

    Implements the edge spanning (Eqs. 14-16) and the removal of one copy of the
    target (Eq. 20). Returns a tuple of (positions, weight):
        positions : the M-1 member positions (with repetition) whose vectors are multiplied
        weight    : the number of distinct orderings of those positions (all give the same product)
    """
    patterns = []
    for slots in combinations_with_replacement(range(c), M):
        counts = Counter(slots)
        if len(counts) < c:  # every member must appear at least once
            continue
        counts[p] -= 1  # remove one copy of the target
        positions = tuple(counts.elements())
        weight = factorial(M - 1) // prod(factorial(k) for k in counts.values())
        patterns.append((positions, weight))
    return tuple(patterns)


def choose_members(
    others: NDArray,
    targets: NDArray,
    k: int,
    strategy: str = "random",
    scores: Optional[NDArray] = None,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Pick k of the other members for every (target, oversized edge) row.

    others  : (n, m) other members of the edge, one row per target (m > k)
    targets : (n,)   the target node of each row
    strategy:
        "random"   - uniform without replacement (what the paper / original code does)
        "topk"     - the k members with the highest scores[target, member]
        "weighted" - sampled without replacement with p ~ exp(score / temperature)
                (Gumbel-top-k trick)
    """
    n, m = others.shape
    if strategy == "random":
        keys = rng.random((n, m))
    elif strategy in ("topk", "weighted"):
        if scores is None:
            raise ValueError(f"strategy '{strategy}' needs a (N, N) `scores` matrix")
        s = scores[targets[:, None], others]
        if strategy == "topk":
            keys = -s
        else:
            keys = -(s / temperature + rng.gumbel(size=(n, m)))
    else:
        raise ValueError(f"unknown strategy '{strategy}'")
    if k <= 4:  # a few argmin passes are much faster than argpartition for small k
        picked = np.empty((n, k), dtype=np.int64)
        rows = np.arange(n)
        for j in range(k):
            picked[:, j] = keys.argmin(axis=1)
            keys[rows, picked[:, j]] = np.inf  # keys is a fresh array, safe to modify
    else:
        picked = np.argpartition(keys, k - 1, axis=1)[:, :k]  # k smallest keys
    return np.take_along_axis(others, picked, axis=1) 


def cosine_basis(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """c_f(k) = 2 cos(2 pi f (k+1) / (2N+1)) for nodes-as-slices k = 0..N-1 and frequencies f = 0..N -> (N, N+1).

    This is the (real) DFT of the symmetrized sequence [0, s_1..s_N, s_N..s_1]; using it
    replaces fft/ifft and keeps everything real.
    """
    Ns = 2 * N + 1
    k = torch.arange(1, N + 1, device=device, dtype=torch.float64)
    f = torch.arange(N + 1, device=device, dtype=torch.float64)
    return (2 * torch.cos(2 * math.pi * torch.outer(k, f) / Ns)).to(dtype)


def adjacency_spectrum(batch: SpectralBatch, N: int, C: torch.Tensor) -> torch.Tensor:
    """A_hat[b, i, j, f] = sum_k A[b, i, j, k] c_f(k), built directly from the terms -> (B, N, N, N+1).

    Each term (target t, partners (a, b), coef) is the adjacency entries a[t,a,b] = a[t,b,a] = coef/2.
    """
    B, Nf = batch.num_graphs, C.shape[1]
    t, a, b = batch.target, batch.idx[:, 0], batch.idx[:, 1]
    w = (batch.coef / 2).to(C.dtype).unsqueeze(1)
    out = C.new_zeros(B * N * N, Nf)
    base = (batch.graph * N + t) * N
    out.index_add_(0, base + a, w * C[b])  # entry a[t, a, b]: row j = a, slice k = b
    out.index_add_(0, base + b, w * C[a])  # entry a[t, b, a]: row j = b, slice k = a
    return out.view(B, N, N, Nf)


def adjacency_tensor(batch: SpectralBatch, N: int, dtype=torch.float32) -> torch.Tensor:
    """Dense degree-normalized adjacency tensor a[b, i, j, k] from the compiled terms -> (B, N, N, N).

    Each term (target t, partners (p, q), coef) is a[t, p, q] = a[t, q, p] = coef / 2.
    """
    B = batch.num_graphs
    t, p, q = batch.target, batch.idx[:, 0], batch.idx[:, 1]
    w = (batch.coef / 2).to(dtype)
    base = (batch.graph * N + t) * N
    A = torch.zeros(B * N * N * N, dtype=dtype, device=batch.x.device)
    A.index_add_(0, (base + p) * N + q, w)
    A.index_add_(0, (base + q) * N + p, w)
    return A.view(B, N, N, N)


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
