import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tmphn import (
    TMPHN,
    HyperGraphLoader,
    build_hypergraph,
    hyperedges_from_adjacency,
    knn_hyperedges,
)


@dataclass
class Config:
    # model / optimization
    Mlst: List[int] = field(default_factory=lambda: [3, 3])  # one order per layer
    hid_dim: int = 64
    epochs: int = 100
    lr: float = 1e-3
    wd: float = 5e-4
    dropout: float = 0.5
    batch_size: int = 64
    eval_batch_size: Optional[int] = None  # None = batch_size
    combine: str = "concat"
    pool: str = "mean"  # graph readout: "mean" | "sum" | "max"
    norm: str = "rms"  # "l2" = paper Eq. 26, "none" = original code
    project_input: bool = True
    # data split
    train_ratio: float = 0.7
    valid_ratio: float = 0.1  # test = the rest
    # how each subject's graph becomes a hypergraph
    hyperedges: str = "edges"  # "edges": every graph edge {u, v}
    # "neighborhood": every node + its neighbors
    # "knn": every node + its knn_k most similar nodes
    knn_k: int = 4
    strategy: str = (
        "random"  # down-sampling of hyperedges larger than M: "random" | "topk" | "weighted"
    )
    scores: str = "auto"  # ranking for knn/topk/weighted: "auto" = |x| if x is square
    # (FC rows as features, e.g. NeuroGraph), else |corr(x)|
    self_loop: bool = False
    # memory / speed
    chunk_size: Optional[int] = None  # see tmphn.layers.tensor_message
    checkpoint: bool = False
    keep_on_device: bool = (
        False  # move all subjects to the GPU once (small datasets only)
    )
    # misc
    seed: int = 42  # one fixed split + initialization, for reproducibility only
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10  # 0 = silent


# =============================================================================
# PyG graphs -> T-MPHN hypergraphs
# =============================================================================
def is_classification(dataset) -> bool:
    y = dataset.y.view(-1).float()
    return bool(torch.equal(y, y.round()))


def graph_edges(edge_index: np.ndarray) -> List[np.ndarray]:
    """Unique undirected edges {u, v} (u != v) as size-2 hyperedges."""
    pairs = np.sort(edge_index.T, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return list(np.unique(pairs, axis=0))


def node_scores(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "features" or (mode == "auto" and x.shape[0] == x.shape[1]):
        return np.abs(x)  # node features are FC rows -> |FC|
    return np.nan_to_num(np.abs(np.corrcoef(x)))


def to_hypergraphs(dataset, cfg: Config):
    needs_scores = cfg.hyperedges == "knn" or cfg.strategy != "random"
    graphs, start = [], time.perf_counter()
    for i, data in enumerate(dataset):
        x = data.x.numpy()
        n = x.shape[0]
        ei = data.edge_index.numpy()
        sim = node_scores(x, cfg.scores) if needs_scores else None
        if cfg.hyperedges == "edges":
            edges = graph_edges(ei)
        elif cfg.hyperedges == "neighborhood":
            A = np.zeros((n, n), dtype=bool)
            A[ei[0], ei[1]] = True
            A[ei[1], ei[0]] = True
            edges = hyperedges_from_adjacency(A)
        elif cfg.hyperedges == "knn":
            edges = knn_hyperedges(sim, cfg.knn_k)
        else:
            raise ValueError(f"unknown hyperedges mode '{cfg.hyperedges}'")
        graphs.append(
            build_hypergraph(
                x,
                edges,
                cfg.Mlst,
                y=int(data.y.view(-1)[0]),
                self_loops=cfg.self_loop,
                strategy=cfg.strategy,
                scores=sim,
                seed=cfg.seed + i,
            )
        )
        if cfg.log_every and (i + 1) % 500 == 0:
            rate = (time.perf_counter() - start) / (i + 1)
            print(
                f"  converted {i + 1}/{len(dataset)} subjects "
                f"(~{rate * (len(dataset) - i - 1):.0f}s left)"
            )
    return graphs


def stratified_split(graphs, cfg: Config):
    labels = np.array([int(g.y) for g in graphs])
    idx = np.arange(len(graphs))
    train, rest = train_test_split(
        idx, train_size=cfg.train_ratio, stratify=labels, random_state=cfg.seed
    )
    val_share = cfg.valid_ratio / (1 - cfg.train_ratio)
    valid, test = train_test_split(
        rest, train_size=val_share, stratify=labels[rest], random_state=cfg.seed
    )
    return {
        name: [graphs[i] for i in ids]
        for name, ids in (("train", train), ("valid", valid), ("test", test))
    }


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(logp: torch.Tensor, y: torch.Tensor) -> dict:
    prob = logp.exp().numpy()
    pred = prob.argmax(1)
    yt = y.numpy()
    num_classes = prob.shape[1]
    m = {
        "loss": F.nll_loss(logp, y).item(),
        "accuracy": accuracy_score(yt, pred),
        "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
        "precision": precision_score(yt, pred, average="macro", zero_division=0),
        "recall": recall_score(yt, pred, average="macro", zero_division=0),
    }
    try:
        if num_classes == 2:
            m["roc_auc"] = roc_auc_score(yt, prob[:, 1])
        else:
            m["roc_auc"] = roc_auc_score(
                yt, prob, multi_class="ovr", labels=list(range(num_classes))
            )
    except ValueError:  # e.g. a class missing from the split
        m["roc_auc"] = float("nan")
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        m["sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
        m["specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    return m


@torch.no_grad()
def predict(model, graphs, device, batch_size):
    model.eval()
    outs, ys = [], []
    for b in HyperGraphLoader(graphs, batch_size, shuffle=False):
        b = b.to(device, non_blocking=True)
        outs.append(model(b.x, b.terms, b.batch, b.num_graphs).cpu())
        ys.append(b.y.cpu())
    return torch.cat(outs), torch.cat(ys)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()  # so timings include the GPU work


# =============================================================================
# Training: one row of metrics per epoch
# =============================================================================
def train_with_history(
    splits, num_features: int, num_classes: int, cfg: Config
) -> pd.DataFrame:
    torch.manual_seed(cfg.seed)
    model = TMPHN(
        num_features,
        cfg.hid_dim,
        num_classes,
        cfg.Mlst,
        combine=cfg.combine,
        dropout=cfg.dropout,
        norm=cfg.norm,
        task="graph",
        pool=cfg.pool,
        chunk_size=cfg.chunk_size,
        checkpoint=cfg.checkpoint,
        project_input=cfg.project_input,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    loader = HyperGraphLoader(
        splits["train"], cfg.batch_size, shuffle=True, seed=cfg.seed
    )
    eval_bs = cfg.eval_batch_size or cfg.batch_size

    rows, cumulative = [], 0.0
    for epoch in range(1, cfg.epochs + 1):
        _sync(cfg.device)
        t0 = time.perf_counter()
        model.train()
        for b in loader:
            b = b.to(cfg.device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.nll_loss(model(b.x, b.terms, b.batch, b.num_graphs), b.y)
            loss.backward()
            optimizer.step()
        _sync(cfg.device)
        train_s = time.perf_counter() - t0
        cumulative += train_s

        t1 = time.perf_counter()
        row = {
            "epoch": epoch,
            "train_time_s": train_s,
            "cumulative_train_time_s": cumulative,
        }
        for name, graphs in splits.items():
            logp, y = predict(model, graphs, cfg.device, eval_bs)
            row.update({f"{name}_{k}": v for k, v in compute_metrics(logp, y).items()})
        row["eval_time_s"] = time.perf_counter() - t1
        rows.append(row)

        if cfg.log_every and (epoch == 1 or epoch % cfg.log_every == 0):
            print(
                f"epoch {epoch:3d} | {train_s:.2f}s | loss {row['train_loss']:.4f} | "
                f"acc train {row['train_accuracy']:.3f} valid {row['valid_accuracy']:.3f} "
                f"test {row['test_accuracy']:.3f} | test auc {row['test_roc_auc']:.3f}"
            )
    return pd.DataFrame(rows)


def plot_history(df: pd.DataFrame, title: str, outfile: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "valid", "test"):
        axes[0].plot(df["epoch"], df[f"{split}_loss"], label=split)
        axes[1].plot(df["epoch"], df[f"{split}_accuracy"], label=split)
    axes[0].set(title=f"{title}: loss", xlabel="epoch")
    axes[1].set(title=f"{title}: accuracy", xlabel="epoch")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)


# =============================================================================
# One full experiment
# =============================================================================
def run_experiment(
    dataset, name: str, cfg: Config, outfile=None, plot: bool = True
) -> pd.DataFrame:
    print(f"\n##### {name}: {len(dataset)} subjects | device {cfg.device}")
    t0 = time.perf_counter()
    graphs = to_hypergraphs(dataset, cfg)
    if cfg.keep_on_device:
        graphs = [g.to(cfg.device) for g in graphs]
    convert_s = time.perf_counter() - t0

    num_features = graphs[0].x.shape[1]
    num_classes = max(int(g.y) for g in graphs) + 1
    splits = stratified_split(graphs, cfg)
    terms = [
        np.mean([g.terms[l].num_terms for g in graphs]) for l in range(len(cfg.Mlst))
    ]
    print(
        f"{num_features} features, {num_classes} classes, splits {[len(v) for v in splits.values()]}, "
        f"avg terms/subject per layer {[int(t) for t in terms]}, conversion {convert_s:.1f}s"
    )

    df = train_with_history(splits, num_features, num_classes, cfg)
    df.insert(0, "dataset", name)
    df["conversion_time_s"] = (
        convert_s  # one-time cost, repeated on every row for convenience
    )

    best = df.loc[df["valid_accuracy"].idxmax()]
    print(
        f"best valid epoch {int(best['epoch'])}: test acc {best['test_accuracy']:.3f}, "
        f"test f1 {best['test_f1_macro']:.3f}, test auc {best['test_roc_auc']:.3f} | "
        f"train time {df['train_time_s'].sum():.1f}s total ({df['train_time_s'].mean():.2f}s/epoch)"
    )

    if outfile is not None:
        out = Path(outfile)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out, index=False)
        if plot:
            plot_history(df, name, out.with_suffix(".png"))
        print(f"saved {out}")
    return df
