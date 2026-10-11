"""
Minimal training / evaluation loops (replaces the T-MPHN parts of prepare.py).

Run the synthetic demo (CPU or GPU):   python -m tmphn.train
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .data import HyperGraph, HyperGraphLoader, build_hypergraph
from .model import TMPHN
from .structure import knn_hyperedges


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def random_split(num_items: int, train: float = 0.5, valid: float = 0.25, seed: int = 0) -> Dict[str, torch.Tensor]:
    perm = torch.from_numpy(np.random.default_rng(seed).permutation(num_items))
    n_train, n_valid = int(train * num_items), int(valid * num_items)
    return {
        "train": perm[:n_train],
        "valid": perm[n_train : n_train + n_valid],
        "test": perm[n_train + n_valid :],
    }


# =============================================================================
# Node classification on one hypergraph (paper's setting)
# =============================================================================
@torch.no_grad()
def evaluate_nodes(model: TMPHN, graph: HyperGraph, split: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model.eval()
    out = model(graph.x, graph.terms)  # one forward pass serves every split
    metrics = {}
    for name, idx in split.items():
        metrics[f"{name}_loss"] = F.nll_loss(out[idx], graph.y[idx]).item()
        metrics[f"{name}_acc"] = (out[idx].argmax(-1) == graph.y[idx]).float().mean().item()
    return metrics


def train_node_classification(
    model: TMPHN,
    graph: HyperGraph,
    split: Dict[str, torch.Tensor],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: Optional[int] = 50,
    device: Optional[str] = None,
    log_every: int = 10,
) -> Dict[str, float]:
    """Full-batch training; restores the weights with the best validation accuracy.

    split: {"train": idx, "valid": idx, "test": idx} (as returned by random_split or
    prepare.rand_train_test_idx).
    """
    device = device or default_device()
    model, graph = model.to(device), graph.to(device)
    split = {k: torch.as_tensor(v, device=device) for k, v in split.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc, best_state, bad_epochs = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.terms)
        loss = F.nll_loss(out[split["train"]], graph.y[split["train"]])
        loss.backward()
        optimizer.step()

        metrics = evaluate_nodes(model, graph, split)
        if metrics["valid_acc"] > best_acc:
            best_acc, best_state, bad_epochs = metrics["valid_acc"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"train {metrics['train_acc']:.3f} | valid {metrics['valid_acc']:.3f}")
        if patience and bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return evaluate_nodes(model, graph, split)


# =============================================================================
# Graph classification over many hypergraphs (e.g. one per subject)
# =============================================================================
@torch.no_grad()
def evaluate_graphs(model: TMPHN, graphs: Sequence[HyperGraph], batch_size: int = 64,
                    device: Optional[str] = None) -> Dict[str, float]:
    device = device or default_device()
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for b in HyperGraphLoader(graphs, batch_size, shuffle=False):
        b = b.to(device)
        out = model(b.x, b.terms, b.batch, b.num_graphs)
        loss_sum += F.nll_loss(out, b.y, reduction="sum").item()
        correct += (out.argmax(-1) == b.y).sum().item()
        total += b.num_graphs
    return {"loss": loss_sum / total, "acc": correct / total}


def train_graph_classification(
    model: TMPHN,
    train_graphs: Sequence[HyperGraph],
    valid_graphs: Sequence[HyperGraph],
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: Optional[int] = 30,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 5,
) -> Dict[str, float]:
    """Mini-batch training; restores the weights with the best validation accuracy."""
    device = device or default_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = HyperGraphLoader(train_graphs, batch_size, shuffle=True, seed=seed)

    best_acc, best_state, bad_epochs = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for b in loader:
            b = b.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.nll_loss(model(b.x, b.terms, b.batch, b.num_graphs), b.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * b.num_graphs

        valid = evaluate_graphs(model, valid_graphs, device=device)
        if valid["acc"] > best_acc:
            best_acc, best_state, bad_epochs = valid["acc"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d} | train loss {epoch_loss / len(train_graphs):.4f} | "
                  f"valid acc {valid['acc']:.3f}")
        if patience and bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return evaluate_graphs(model, valid_graphs, device=device)


# =============================================================================
# Synthetic demo: exercises both tasks, down-sampling, and batching
# =============================================================================
def _demo() -> None:
    device = default_device()
    rng = np.random.default_rng(0)
    print(f"device: {device}")

    # --- node classification: one hypergraph, labels depend on features ---
    N, F_in = 300, 16
    x = rng.normal(size=(N, F_in)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    edges = [rng.choice(N, size=int(rng.integers(2, 9)), replace=False) for _ in range(400)]
    graph = build_hypergraph(x, edges, Mlst=[4, 3], y=y, self_loops=True, seed=0)
    print(graph)
    model = TMPHN(F_in, 64, 2, Mlst=[4, 3], dropout=0.1)
    print("node task:", train_node_classification(model, graph, random_split(N), epochs=60, log_every=20))

    # --- graph classification: many small hypergraphs, kNN hyperedges, top-k down-sampling ---
    graphs = []
    for i in range(80):
        n = int(rng.integers(20, 40))
        xi = rng.normal(size=(n, F_in)).astype(np.float32)
        label = i % 2
        xi[:, 0] += 1.0 if label else -1.0  # a learnable signal
        sim = xi @ xi.T
        graphs.append(build_hypergraph(xi, knn_hyperedges(sim, k=4), Mlst=[5, 3], y=label,
                                       strategy="topk", scores=sim))
    model = TMPHN(F_in, 64, 2, Mlst=[5, 3], task="graph", dropout=0.1)
    print("graph task:", train_graph_classification(model, graphs[:60], graphs[60:], epochs=30, log_every=10))


if __name__ == "__main__":
    _demo()
