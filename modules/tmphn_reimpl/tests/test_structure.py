"""
NumPy-only tests for tmphn/structure.py (no PyTorch needed).

Run:  python tests/test_structure.py      (or: pytest tests/test_structure.py)
"""
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np

# Load structure.py directly so these tests run even where torch is not installed.
_path = Path(__file__).resolve().parents[1] / "tmphn" / "structure.py"
_spec = importlib.util.spec_from_file_location("tmphn_structure", _path)
S = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = S  # dataclasses need the module to be registered
_spec.loader.exec_module(S)


# -----------------------------------------------------------------------------
# Reference: line-by-line port of the ORIGINAL TMessagePassing logic
# (aggregate_for_one_node / aggregate_with_M / aggregate_with_c / adj_coef)
# -----------------------------------------------------------------------------
def reference_messages(X, edges, M):
    N, D = X.shape
    table = {v: [[int(u) for u in e] for e in edges if v in e] for v in range(N)}
    out = np.zeros((N, D))
    for v in range(N):
        es = table[v]
        if not es:  # original returned self features here; new code returns 0
            continue
        for edge in es:
            c = len(edge)
            alpha = sum(((-1) ** i) * math.comb(c, i) * ((c - i) ** M) for i in range(c))
            adj = (c / alpha) / len(es)
            if c == M:
                rest = list(edge)
                rest.remove(v)
                out[v] += adj * math.factorial(M - 1) * np.prod(X[rest], axis=0)
            elif c < M:
                all_comb = [list(t) for t in itertools.combinations_with_replacement(edge, M)]
                val_comb = [cmb for cmb in all_comb if set(cmb) == set(edge)]
                for cmb in val_comb:
                    cmb.remove(v)
                    perms = len(set(itertools.permutations(cmb)))
                    out[v] += adj * perms * np.prod(X[cmb], axis=0)
    return out


def compiled_messages(X, ct):
    """NumPy mirror of layers.tensor_message: gather, Hadamard product, scatter-add."""
    out = np.zeros_like(X, dtype=np.float64)
    contrib = np.prod(X[ct.idx], axis=1) * ct.coef[:, None]
    np.add.at(out, ct.target, contrib)
    return out


def random_hypergraph(rng, N, num_edges, max_size, min_size=1):
    return [rng.choice(N, size=int(rng.integers(min_size, max_size + 1)), replace=False) for _ in range(num_edges)]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_matches_original_when_no_sampling():
    rng = np.random.default_rng(0)
    for M in (2, 3, 4, 5):
        N, D = 25, 6
        edges = random_hypergraph(rng, N, 40, max_size=M)  # all |e| <= M, incl. singletons
        X = rng.normal(size=(N, D))
        ref = reference_messages(X, edges, M)
        new = compiled_messages(X, S.compile_terms(edges, N, M))
        assert np.allclose(ref, new, rtol=1e-5, atol=1e-6), f"mismatch for M={M}"


def test_paper_example_fig2():
    # e1 = {v1, v2}, e2 = {v1, v2, v3}, e3 = {v3, v4, v5}; D = 1, M = 3.
    # Paper: m(v1) = 1/3 (x1x2 + x2x1 + x2^2) + 1/2 (x2x3 + x3x2); the code also divides by deg(v1) = 2.
    edges = [[0, 1], [0, 1, 2], [2, 3, 4]]
    x = np.array([[1.3], [0.7], [-2.0], [0.5], [1.1]])
    x1, x2, x3 = x[0, 0], x[1, 0], x[2, 0]
    expected_v1 = (1 / 3 * (2 * x1 * x2 + x2**2) + 1 / 2 * (2 * x2 * x3)) / 2
    m = compiled_messages(x, S.compile_terms(edges, 5, 3))
    assert np.isclose(m[0, 0], expected_v1)


def test_adjacency_values_match_paper():
    assert np.isclose(S.adjacency_value(2, 3), 1 / 3)  # Fig. 5(c)
    assert np.isclose(S.adjacency_value(3, 3), 1 / 2)
    for M in range(2, 7):  # a_M * (M-1)! == 1 for a full-size edge
        assert np.isclose(S.adjacency_value(M, M) * math.factorial(M - 1), 1.0)


def test_isolated_nodes_get_no_terms():
    ct = S.compile_terms([[0, 1]], num_nodes=4, M=3)
    assert set(ct.target.tolist()) == {0, 1}


def test_oversized_edges_are_downsampled_not_skipped():
    big = np.arange(10)
    ct = S.compile_terms([big], num_nodes=10, M=3, seed=0)
    assert ct.num_terms == 10  # one term per member (the original code skipped these edges)
    for t, row, w in zip(ct.target, ct.idx, ct.coef):
        assert len(set(row.tolist())) == 2 and t not in row and set(row) <= set(big)
        assert np.isclose(w, 1.0)  # 1 / degree, degree = 1


def test_sampling_is_seeded_and_topk_uses_scores():
    edges = [np.arange(12)]
    a = S.compile_terms(edges, 12, 4, seed=1).idx
    b = S.compile_terms(edges, 12, 4, seed=1).idx
    c = S.compile_terms(edges, 12, 4, seed=2).idx
    assert np.array_equal(a, b) and not np.array_equal(a, c)

    scores = np.random.default_rng(3).random((12, 12))
    ct = S.compile_terms(edges, 12, 4, strategy="topk", scores=scores)
    for t, row in zip(ct.target, ct.idx):
        others = [u for u in range(12) if u != t]
        best = sorted(others, key=lambda u: -scores[t, u])[:3]
        assert set(row.tolist()) == set(best)

    w = S.compile_terms(edges, 12, 4, strategy="weighted", scores=scores, temperature=0.1, seed=0)
    for t, row in zip(w.target, w.idx):
        assert len(set(row.tolist())) == 3 and t not in row


def test_builders():
    H = np.array([[1, 0], [1, 1], [0, 1]])
    assert [e.tolist() for e in S.hyperedges_from_incidence(H)] == [[0, 1], [1, 2]]
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert [e.tolist() for e in S.hyperedges_from_adjacency(A)] == [[0, 1], [0, 1, 2], [1, 2]]
    sim = np.array([[1, 0.9, 0.1], [0.9, 1, 0.2], [0.1, 0.2, 1]])
    assert [e.tolist() for e in S.knn_hyperedges(sim, 1)] == [[0, 1], [1, 0], [2, 1]]
    kept = S.induced_hyperedges([[0, 1, 2], [2]], nodes=[0, 1], num_nodes=3)
    assert [e.tolist() for e in kept] == [[0, 1]]
    loops = S.add_self_loops([[0, 1]], 2)
    assert [e.tolist() for e in loops] == [[0, 1], [0], [1]]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\nall {len(tests)} tests passed")
