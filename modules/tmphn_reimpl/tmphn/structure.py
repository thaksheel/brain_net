"""
Hypergraph structure for T-MPHN (NumPy only, runs once per hypergraph).

Replaces: Neighbors.py (NeighborFinder) and the combinatorics inside
TMessagePassing (aggregate_with_M, aggregate_with_c, adj_coef).

Pipeline:   hyperedges  --compile_terms(M)-->  CompiledTerms

A *term* is one summand of the T-MPHN neighborhood message (paper Eqs. 21-24):

    message[target] += coef * h[idx[0]] * h[idx[1]] * ... * h[idx[M-2]]

i.e. the Hadamard (cross-node interaction) product of M-1 node vectors, weighted
by the adjacency value of its hyperedge divided by the target's degree.
Compiling every node's neighborhood into flat arrays ONCE lets the model run each
layer as a single gather / multiply / scatter-add on the GPU (see layers.py).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb, factorial, prod
from typing import List, Optional, Sequence

import numpy as np

Hyperedges = List[np.ndarray]  # each entry: the node ids of one hyperedge
_MAX_CHUNK_ELEMENTS = 4_000_000  # temporary array size when down-sampling large hyperedges


# =============================================================================
# 1. Building hyperedge lists
# =============================================================================
def hyperedges_from_incidence(H) -> Hyperedges:
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


def hyperedges_from_adjacency(A) -> Hyperedges:
    """One hyperedge per node: the node together with its graph neighbors."""
    A = (np.asarray(A) != 0) | np.eye(len(A), dtype=bool)
    rows, cols = np.nonzero(A)  # row-major, so each row's columns come out sorted
    return np.split(cols.astype(np.int64), np.flatnonzero(np.diff(rows)) + 1)


def knn_hyperedges(similarity, k: int) -> Hyperedges:
    """One hyperedge per node: the node plus its k most similar other nodes.

    This is the construction used for the paper's 3D-object datasets (K = 5).
    For brain data pass |FC| so each ROI is grouped with its k most strongly
    connected ROIs.
    """
    S = np.array(similarity, dtype=np.float64)  # copy
    np.fill_diagonal(S, -np.inf)
    nearest = np.argsort(-S, axis=1)[:, :k]
    return [np.concatenate(([i], nearest[i])).astype(np.int64) for i in range(S.shape[0])]


def add_self_loops(hyperedges: Sequence, num_nodes: int) -> Hyperedges:
    """Append a singleton hyperedge {v} for every node (same as prepare.add_self_loop)."""
    return [np.asarray(e, dtype=np.int64) for e in hyperedges] + [
        np.array([v], dtype=np.int64) for v in range(num_nodes)
    ]


def induced_hyperedges(hyperedges: Sequence, nodes, num_nodes: int) -> Hyperedges:
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


# =============================================================================
# 3. Down-sampling hyperedges that are larger than M
# =============================================================================
def choose_members(
    others: np.ndarray,
    targets: np.ndarray,
    k: int,
    strategy: str = "random",
    scores: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
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


# =============================================================================
# 4. Compilation: neighborhood table -> flat term arrays
# =============================================================================
@dataclass(repr=False)
class CompiledTerms:
    target: np.ndarray  # (T,)      int64: node that receives the term
    idx: np.ndarray     # (T, M-1)  int64: nodes whose vectors are multiplied
    coef: np.ndarray    # (T,)      float32: a_e * #orderings / degree(target)
    num_nodes: int
    M: int

    @property
    def num_terms(self) -> int:
        return int(self.target.shape[0])

    def __repr__(self) -> str:
        return f"CompiledTerms(M={self.M}, num_nodes={self.num_nodes}, num_terms={self.num_terms})"


def compile_terms(
    hyperedges: Sequence,
    num_nodes: int,
    M: int,
    strategy: str = "random",
    scores: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    seed=None,
) -> CompiledTerms:
    """Compile the order-M neighborhood table of a hypergraph into flat term arrays.

    For every node v and every hyperedge e containing v (degree d_v = #edges containing v):
      |e| <= M : the spanned products of Eqs. 14-16 / 23 with coef a_e * weight / d_v
                 (for |e| == M this is a single product of the other M-1 members, coef 1 / d_v)
      |e| >  M : e is down-sampled to v plus M-1 chosen members (paper Sec. IV-D);
                 a sampled M-node edge has coef a_M * (M-1)! / d_v = 1 / d_v
    Nodes in no hyperedge get no terms (their message is zero).

    `seed` may be an int, None, or a numpy Generator (only used by "random"/"weighted").
    """
    if M < 2:
        raise ValueError("M must be at least 2")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32)  # only used for ranking members

    edges = [np.unique(np.asarray(e, dtype=np.int64)) for e in hyperedges]
    edges = [e for e in edges if e.size > 0]
    degree = np.zeros(num_nodes, dtype=np.int64)
    if edges:
        all_members = np.concatenate(edges)
        if all_members.min() < 0 or all_members.max() >= num_nodes:
            raise ValueError("hyperedge contains a node id outside [0, num_nodes)")
        np.add.at(degree, all_members, 1)

    targets, idxs, coefs = [], [], []
    sizes = np.array([e.size for e in edges], dtype=np.int64)
    for c in np.unique(sizes):
        c = int(c)
        E = np.stack([edges[i] for i in np.flatnonzero(sizes == c)])  # (n_c, c)
        if c <= M:
            a = adjacency_value(c, M)
            for p in range(c):  # every member takes a turn as the target
                tgt = E[:, p]
                for positions, weight in expansion_patterns(c, M, p):
                    targets.append(tgt)
                    idxs.append(E[:, list(positions)])
                    coefs.append(a * weight / degree[tgt])
        else:
            # every (edge, member) pair at once: row p of `others` = the members other than p
            not_self = ~np.eye(c, dtype=bool)
            rows_per_chunk = max(1, _MAX_CHUNK_ELEMENTS // (c * c))  # caps temporary memory
            for start in range(0, len(E), rows_per_chunk):
                Ec = E[start : start + rows_per_chunk]
                n = Ec.shape[0]
                tgt = Ec.reshape(-1)                                                   # (n*c,)
                others = np.broadcast_to(Ec[:, None, :], (n, c, c))[:, not_self].reshape(n * c, c - 1)
                idxs.append(choose_members(others, tgt, M - 1, strategy, scores, temperature, rng))
                targets.append(tgt)
                coefs.append(1.0 / degree[tgt])

    if targets:
        target = np.ascontiguousarray(np.concatenate(targets), dtype=np.int64)
        idx = np.ascontiguousarray(np.concatenate(idxs), dtype=np.int64)
        coef = np.ascontiguousarray(np.concatenate(coefs), dtype=np.float32)
    else:
        target = np.zeros(0, dtype=np.int64)
        idx = np.zeros((0, M - 1), dtype=np.int64)
        coef = np.zeros(0, dtype=np.float32)
    return CompiledTerms(target, idx, coef, num_nodes, M)
