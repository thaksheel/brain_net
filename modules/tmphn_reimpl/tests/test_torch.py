"""
PyTorch tests. Run from the repository root:

    python tests/test_torch.py                                       # new code only
    python tests/test_torch.py path/to/original/tmessagepassing.py   # + compare with the original aggregator

(pytest also works; set TMPHN_ORIGINAL=path/to/tmessagepassing.py for the comparison.)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tmphn import TMPHN, Terms, build_hypergraph, collate, compile_terms, tensor_message  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REF = _load("tmphn_test_structure", ROOT / "tests" / "test_structure.py")  # reference_messages, random_hypergraph


def _toy(seed=0, N=30, D=8, M=4, num_edges=50):
    rng = np.random.default_rng(seed)
    return REF.random_hypergraph(rng, N, num_edges, max_size=M), rng.normal(size=(N, D))


def test_message_matches_reference():
    for M in (2, 3, 4, 5):
        edges, X = _toy(M=M)
        terms = Terms.from_compiled(compile_terms(edges, X.shape[0], M))
        new = tensor_message(torch.tensor(X, dtype=torch.float64), terms).numpy()
        assert np.allclose(REF.reference_messages(X, edges, M), new, rtol=1e-5, atol=1e-6), f"M={M}"


def test_message_matches_original():
    path = os.environ.get("TMPHN_ORIGINAL")
    if not path:
        print("      (skipped: pass the original tmessagepassing.py path to run)")
        return
    original = _load("original_tmessagepassing", path)
    M = 4
    edges, X = _toy(M=M)  # all |e| <= M, so no sampling differences
    N = X.shape[0]
    Xt = torch.tensor(X, dtype=torch.float32)
    table = {v: [[int(u) for u in e] for e in edges if v in e] for v in range(N)}
    agg = original.TMessagePassing(lambda idx: Xt[idx], table, M, torch.device("cpu"))
    with torch.no_grad():
        ref = agg(torch.arange(N))
    new = tensor_message(Xt, Terms.from_compiled(compile_terms(edges, N, M)))
    has_edges = torch.tensor([len(table[v]) > 0 for v in range(N)])  # original returns x_v for isolated nodes
    assert torch.allclose(ref[has_edges], new[has_edges], rtol=1e-4, atol=1e-5)


def test_chunking_and_checkpoint_match():
    edges, X = _toy(N=40, num_edges=80)
    terms = Terms.from_compiled(compile_terms(edges, 40, 4))

    def run(**kwargs):
        h = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        m = tensor_message(h, terms, **kwargs)
        m.pow(2).sum().backward()
        return m.detach(), h.grad

    m0, g0 = run()
    for kwargs in ({"chunk_size": 7}, {"checkpoint": True}, {"chunk_size": 7, "checkpoint": True}):
        m1, g1 = run(**kwargs)
        assert torch.allclose(m0, m1) and torch.allclose(g0, g1), kwargs


def test_batching_matches_single_graphs():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    graphs = []
    for i in range(5):
        n = int(rng.integers(8, 15))
        edges = REF.random_hypergraph(rng, n, 12, max_size=6, min_size=2)  # some edges > M -> sampled
        graphs.append(build_hypergraph(rng.normal(size=(n, 6)), edges, Mlst=[4, 3], y=i % 2, seed=i))
    model = TMPHN(6, 16, 2, Mlst=[4, 3], task="graph").eval()
    b = collate(graphs)
    with torch.no_grad():
        batched = model(b.x, b.terms, b.batch, b.num_graphs)
        single = torch.cat([model(g.x, g.terms) for g in graphs])
    assert torch.allclose(batched, single, atol=1e-5)


def test_training_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N = 120
    x = rng.normal(size=(N, 8))
    y = (x[:, 0] > 0).astype(np.int64)
    edges = REF.random_hypergraph(rng, N, 150, max_size=6, min_size=2)
    g = build_hypergraph(x, edges, Mlst=[4, 3], y=y, self_loops=True)
    model = TMPHN(8, 32, 2, Mlst=[4, 3], dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.nll_loss(model(g.x, g.terms), g.y)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0], losses


def test_state_dict_has_no_aliases():
    model = TMPHN(8, 16, 3, Mlst=[5, 4, 3])
    assert len(model.state_dict()) == len(list(model.parameters()))


def test_wrong_M_is_rejected():
    edges, X = _toy()
    g = build_hypergraph(X, edges, Mlst=[4])
    try:
        TMPHN(X.shape[1], 16, 2, Mlst=[3])(g.x, g.terms)
    except ValueError:
        return
    raise AssertionError("terms compiled with M=4 were accepted by a layer with M=3")


def test_cuda_matches_cpu():
    if not torch.cuda.is_available():
        print("      (skipped: no CUDA device)")
        return
    torch.manual_seed(0)
    edges, X = _toy(N=50, num_edges=100)
    g = build_hypergraph(X, edges, Mlst=[4, 3])
    model = TMPHN(X.shape[1], 32, 2, Mlst=[4, 3]).eval()
    with torch.no_grad():
        cpu = model(g.x, g.terms)
        gc = g.to("cuda")
        gpu = model.cuda()(gc.x, gc.terms).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["TMPHN_ORIGINAL"] = sys.argv[1]
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\nall {len(tests)} tests passed")
