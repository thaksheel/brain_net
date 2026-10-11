"""
Graph classification (one hypergraph per subject) with the new T-MPHN implementation.
Records metrics and runtime for EVERY epoch, for every atlas, and writes one Excel file per atlas.

Uses your existing data pipeline (src.Preprocess / src.GraphDataset) to load the ADHD data,
then converts each subject's graph to a hypergraph for T-MPHN.

Put this file and the `tmphn/` folder in your project root (next to `src/`), then:
    python run_graph_classification.py
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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

from datasets import GraphDataset 
from preprocess import Preprocess
from tmphn import (
    TMPHN,
    HyperGraphLoader,
    build_hypergraph,
    hyperedges_from_adjacency,
    knn_hyperedges,
)


@dataclass
class Config:
    Mlst: List[int] = field(
        default_factory=lambda: [3, 3, 3]
    )  # one order per layer -> 2 layers
    hid_dim: int = 128
    epochs: int = 400
    lr: float = 5e-3
    wd: float = 5e-3
    dropout: float = 0.5
    batch_size: int = 128
    train_ratio: float = 0.6
    valid_ratio: float = 0.2  # test = the remaining 0.2
    combine: str = "concat"
    pool: str = "mean"  # graph readout: "mean" | "sum" | "max"
    norm: str = "rms"  # "l2" = paper Eq. 26, "none" = original code
    project_input: bool = True
    hyperedges: str = "edges"  # "edges": each graph edge {u, v} (natural for M = 2)
    # "neighborhood": each node + its neighbors (use M >= 3)
    # "knn": each node + its k most similar nodes
    knn_k: int = 4
    strategy: str = (
        "random"  # down-sampling of hyperedges larger than M: "random" | "topk" | "weighted"
    )
    self_loop: bool = False
    seed: int = 42  # one fixed split + initialization, only for reproducibility
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    display: bool = True


ATLAS = ["AAL116", "PP264", "Schaefer100"]
ROIS = [116, 264, 100]
DATA_DIR = "../../data/adhd/"
OUTFILES = [
    "./exports/tmphn_epochs_a116.xlsx",
    "./exports/tmphn_epochs_p264.xlsx",
    "./exports/tmphn_epochs_s100.xlsx",
]
PLOT = True  # also save learning curves next to each Excel file


# =============================================================================
# PyG graphs -> T-MPHN hypergraphs
# =============================================================================
def graph_edges(edge_index: np.ndarray) -> List[np.ndarray]:
    """Unique undirected edges {u, v} (u != v) as size-2 hyperedges."""
    pairs = np.sort(edge_index.T, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return list(np.unique(pairs, axis=0))


def node_similarity(x: np.ndarray) -> np.ndarray:
    """|Pearson correlation| between node feature vectors (used by "knn" and "topk"/"weighted")."""
    return np.nan_to_num(np.abs(np.corrcoef(x)))


def to_hypergraphs(dataset, cfg: Config):
    graphs = []
    for data in dataset:
        x = data.x.numpy()
        n = x.shape[0]
        ei = data.edge_index.numpy()
        sim = (
            node_similarity(x)
            if (cfg.hyperedges == "knn" or cfg.strategy != "random")
            else None
        )
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
                seed=cfg.seed,
            )
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


def compute_metrics(logp: torch.Tensor, y: torch.Tensor) -> dict:
    prob = logp.exp().numpy()
    pred = prob.argmax(1)
    yt = y.numpy()
    m = {
        "loss": F.nll_loss(logp, y).item(),
        "accuracy": accuracy_score(yt, pred),
        "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
        "precision": precision_score(yt, pred, average="macro", zero_division=0),
        "recall": recall_score(yt, pred, average="macro", zero_division=0),
    }
    try:
        m["roc_auc"] = (
            roc_auc_score(yt, prob[:, 1])
            if prob.shape[1] == 2
            else roc_auc_score(yt, prob, multi_class="ovr")
        )
    except ValueError:  # e.g. only one class present in the split
        m["roc_auc"] = float("nan")
    if prob.shape[1] == 2:
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        m["sensitivity"] = tp / (tp + fn) if tp + fn else float("nan")
        m["specificity"] = tn / (tn + fp) if tn + fp else float("nan")
    return m


@torch.no_grad()
def predict(model, graphs, device, batch_size=256):
    model.eval()
    outs, ys = [], []
    for b in HyperGraphLoader(graphs, batch_size, shuffle=False):
        b = b.to(device)
        outs.append(model(b.x, b.terms, b.batch, b.num_graphs).cpu())
        ys.append(b.y.cpu())
    return torch.cat(outs), torch.cat(ys)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()  # so the timings include the GPU work


# =============================================================================
# Training: one row of metrics per epoch
# =============================================================================
def train_with_history(splits, num_features, num_classes, cfg: Config) -> pd.DataFrame:
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
        project_input=cfg.project_input,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    loader = HyperGraphLoader(
        splits["train"], cfg.batch_size, shuffle=True, seed=cfg.seed
    )

    rows, cumulative = [], 0.0
    for epoch in range(1, cfg.epochs + 1):
        _sync(cfg.device)
        t0 = time.perf_counter()
        model.train()
        for b in loader:
            b = b.to(cfg.device)
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
            logp, y = predict(model, graphs, cfg.device)
            row.update({f"{name}_{k}": v for k, v in compute_metrics(logp, y).items()})
        row["eval_time_s"] = time.perf_counter() - t1
        rows.append(row)

        if cfg.display and (epoch == 1 or epoch % 10 == 0):
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
    if outfile:
        fig.savefig(outfile, dpi=120)
    plt.close(fig)


def main():
    cfg = Config()
    for k, atlas in enumerate(ATLAS):
        pp = Preprocess(n_rois=ROIS[k], normalize="minmax")
        nodes, edges, y, errors = pp.from_node_features_to_graph_dataset(
            DATA_DIR, atlas
        )
        data_list = pp.get_graph_dataset_v2(nodes, edges, y)
        dataset = GraphDataset(root=f"{DATA_DIR}{atlas}", data_list=data_list)

        t0 = time.perf_counter()
        graphs = [
            g.to(cfg.device) for g in to_hypergraphs(dataset, cfg)
        ]  # small dataset: keep it on the device
        convert_s = time.perf_counter() - t0
        splits = stratified_split(graphs, cfg)
        print(
            f"\n##### {atlas}: {len(graphs)} subjects, {dataset.num_features} features, "
            f"{dataset.num_classes} classes | splits {[len(v) for v in splits.values()]} | "
            f"hypergraph conversion {convert_s:.2f}s | device {cfg.device}"
        )

        df = train_with_history(splits, dataset.num_features, dataset.num_classes, cfg)
        df.insert(0, "atlas", atlas)

        best = df.loc[df["valid_accuracy"].idxmax()]
        print(
            f"best valid epoch {int(best['epoch'])}: test acc {best['test_accuracy']:.3f}, "
            f"test f1 {best['test_f1_macro']:.3f}, test auc {best['test_roc_auc']:.3f} | "
            f"total train time {df['train_time_s'].sum():.2f}s "
            f"({df['train_time_s'].mean():.3f}s/epoch)"
        )

        # out = Path(OUTFILES[k])
        # out.parent.mkdir(parents=True, exist_ok=True)
        # df.to_excel(out, index=False)
        if PLOT:
            plot_history(df, atlas, None)
        # print(f"saved {out}")
    print("END")


if __name__ == "__main__":
    main()
