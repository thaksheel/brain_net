import torch
from typing import Optional, Sequence
import numpy as np

from .terms import CompiledTerms, Terms
from .spectral import add_self_loops, SpectralGraph
from .hyper import HyperGraph
from .utils import adjacency_value, expansion_patterns, choose_members

_MAX_CHUNK_ELEMENTS = (
    4_000_000  # temporary array size when down-sampling large hyperedges
)


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
            rows_per_chunk = max(
                1, _MAX_CHUNK_ELEMENTS // (c * c)
            )  # caps temporary memory
            for start in range(0, len(E), rows_per_chunk):
                Ec = E[start : start + rows_per_chunk]
                n = Ec.shape[0]
                tgt = Ec.reshape(-1)  # (n*c,)
                others = np.broadcast_to(Ec[:, None, :], (n, c, c))[
                    :, not_self
                ].reshape(n * c, c - 1)
                idxs.append(
                    choose_members(
                        others, tgt, M - 1, strategy, scores, temperature, rng
                    )
                )
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


def build_spectral_graph(
    x,
    hyperedges: Sequence,
    y=None,
    self_loops: bool = False,
    strategy: str = "random",
    scores=None,
    temperature: float = 1.0,
    seed: Optional[int] = 0,
) -> SpectralGraph:
    """Compile one hypergraph for TSpectralHGNN (same arguments as build_hypergraph, M fixed to 3)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    n = int(x.shape[0])
    if self_loops:
        hyperedges = add_self_loops(hyperedges, n)
    ct = compile_terms(
        hyperedges, n, 3, strategy, scores, temperature, np.random.default_rng(seed)
    )
    if y is not None:
        y = torch.as_tensor(np.asarray(y), dtype=torch.long).reshape(-1)
    return SpectralGraph(
        x,
        torch.from_numpy(ct.target),
        torch.from_numpy(ct.idx),
        torch.from_numpy(ct.coef),
        y,
    )


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
