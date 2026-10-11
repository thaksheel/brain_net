"""
Node classification with the new T-MPHN implementation
(counterpart of the old `GraphTrainer.train_node_cls` script).

Expects the same dataset layout as before:
    <DATA_DIR>/<dataset>/hypergraph.pickle   incidence matrix H (N x E), dense or scipy.sparse
    <DATA_DIR>/<dataset>/feature.pickle      node features X (N x D)
    <DATA_DIR>/<dataset>/labels.pickle       integer labels y (N,)

Run from the folder that contains the `tmphn` package:
    python run_node_classification.py
"""
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from tmphn import TMPHN, build_hypergraph, hyperedges_from_incidence, train_node_classification


# =============================================================================
# Settings (same names as your Params where they apply)
# =============================================================================
@dataclass
class Config:
    Mlst: List[int] = field(default_factory=lambda: [5, 5])  # one order per layer -> 2 layers
    hid_dim: int = 128
    epochs: int = 100
    lr: float = 1e-3
    wd: float = 1e-3
    dropout: float = 0.2
    combine: str = "concat"
    self_loop: bool = False
    strategy: str = "random"       # how hyperedges larger than a layer's M are down-sampled
    norm: str = "rms"              # "l2" = paper Eq. 26, "none" = original code
    project_input: bool = True     # False = feed raw X to layer 0, like the original code
    train_ratio: float = 0.5
    valid_ratio: float = 0.25
    patience: Optional[int] = None  # None = run all epochs, keep the best-validation weights
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    display: bool = True


DATA_DIR = Path("./data/academic_net")
DATASETS = ["cora-cc", "cora-ca", "dblp-ca", "citeseer-cc"]
SEEDS = list(range(42, 42 + 1))      # e.g. range(42, 52) for mean +/- std over 10 runs
EXPORT: Optional[Path] = None        # e.g. Path("./exports/tmphn_node_cls.csv")


# =============================================================================
# Data
# =============================================================================
def load_dataset(path: Path):
    with open(path / "hypergraph.pickle", "rb") as f:
        H = pickle.load(f)
    with open(path / "feature.pickle", "rb") as f:
        X = pickle.load(f)
    with open(path / "labels.pickle", "rb") as f:
        y = pickle.load(f)
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)  # dense features
    y = np.asarray(y)
    if y.ndim == 2:  # one-hot labels -> class ids
        y = y.argmax(axis=1)
    return X.astype(np.float32), H, y.astype(np.int64).reshape(-1)


def balanced_split(y: torch.Tensor, seed: int, train_prop: float, valid_prop: float):
    """Same procedure as prepare.rand_train_test_idx(balance=True),
    so the splits match your old runs for the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    per_class = []
    for c in range(int(y.max()) + 1):
        idx = torch.where(y == c)[0].view(-1)
        per_class.append(idx[torch.randperm(idx.size(0))])
    n_train_per_class = int(train_prop / (y.max() + 1) * len(y))
    n_valid = int(valid_prop * len(y))
    train = torch.cat([idx[:n_train_per_class] for idx in per_class])
    rest = torch.cat([idx[n_train_per_class:] for idx in per_class])
    rest = rest[torch.randperm(rest.size(0))]
    return {"train": train, "valid": rest[:n_valid], "test": rest[n_valid:]}


# =============================================================================
# One run = one dataset x one seed
# =============================================================================
@torch.no_grad()
def macro_f1(model: TMPHN, graph, split) -> dict:
    model.eval()
    pred = model(graph.x, graph.terms).argmax(-1).cpu()
    y = graph.y.cpu()
    return {f"{name}_f1_macro": f1_score(y[idx], pred[idx], average="macro") for name, idx in split.items()}


def run_once(X, H, y, cfg: Config, seed: int) -> dict:
    edges = hyperedges_from_incidence(H)
    graph = build_hypergraph(X, edges, cfg.Mlst, y=y, self_loops=cfg.self_loop,
                             strategy=cfg.strategy, seed=seed)
    split = balanced_split(graph.y, seed, cfg.train_ratio, cfg.valid_ratio)

    torch.manual_seed(seed)  # weight initialization
    model = TMPHN(
        in_dim=graph.x.shape[1],
        hid_dim=cfg.hid_dim,
        num_classes=int(graph.y.max()) + 1,  # taken from the labels, not hard-coded
        Mlst=cfg.Mlst,
        combine=cfg.combine,
        dropout=cfg.dropout,
        norm=cfg.norm,
        project_input=cfg.project_input,
    )

    start = time.time()
    metrics = train_node_classification(
        model, graph, split,
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.wd, patience=cfg.patience,
        device=cfg.device, log_every=10 if cfg.display else 0,
    )  # returns metrics with the best-validation weights restored
    metrics["runtime_s"] = time.time() - start
    metrics.update(macro_f1(model, graph.to(cfg.device), split))
    metrics["num_terms"] = "/".join(str(t.num_terms) for t in graph.terms)
    return metrics


# =============================================================================
# Main loop
# =============================================================================
def main():
    cfg = Config()
    print(cfg)
    rows = []
    for name in DATASETS:
        X, H, y = load_dataset(DATA_DIR / name)  # load once per dataset
        print(f"\n##### {name}: {X.shape[0]} nodes, {X.shape[1]} features, {int(y.max()) + 1} classes")
        for seed in SEEDS:
            print(f"\n=== {name} | seed {seed} | device {cfg.device} ===")
            metrics = run_once(X, H, y, cfg, seed)
            rows.append({"dataset": name, "seed": seed, **metrics})
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
            print(f"\n\tRuntime={metrics['runtime_s']:.2f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby("dataset")[["test_acc", "test_f1_macro", "runtime_s"]].agg(["mean", "std"])
    print("\n", summary)
    if EXPORT is not None:
        EXPORT.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(EXPORT, index=False) if EXPORT.suffix == ".xlsx" else df.to_csv(EXPORT, index=False)
        print(f"saved {EXPORT}")
    print("END")


if __name__ == "__main__":
    main()
