from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    APPNP,
    MLP,
    ChebConv,
    GATConv,
    GCNConv,
    GeneralConv,
    GINConv,
    GraphConv,
    SAGEConv,
    SGConv,
    TransformerConv,
)
from tqdm import tqdm

from .base_trainer import TENSOR_METHODS, BaseTrainer, node_scores, set_seed
from .config import TTV
from .gnn import GIN
from .residual_gnn import ResidualGNNs
from .scoring import RunResult
from .thgnn import (
    HyperGraphLoader,
    SpectralGraphLoader,
    build_hypergraph,
    build_spectral_graph,
    hyperedges_from_adjacency,
    knn_hyperedges,
)
from .utils import NeuroGraphDataset

NG_MODELS = {
    "GCNConv": GCNConv,
    "GINConv": GINConv,
    "GraphConv": GraphConv,
    "TransformerConv": TransformerConv,
    "GATConv": GATConv,
    "GeneralConv": GeneralConv,
    "SGConv": SGConv,
    "ChebConv": ChebConv,
    "SAGEConv": SAGEConv,
    "APPNP": APPNP,
    "MLP": MLP,
}
GRAPH_METHODS = TENSOR_METHODS + ("GIN",) + tuple(NG_MODELS)


class GraphTrainer(BaseTrainer):
    """Graph classification: one brain network per subject, methods in GRAPH_METHODS."""

    def load_dataset(self, root: str, name: str) -> NeuroGraphDataset:
        dataset = NeuroGraphDataset(root=root, name=name)
        self.params.dataset = name
        self.params.num_features = dataset.num_features
        self.params.num_classes = dataset.num_classes
        return dataset

    def labels(self, dataset) -> np.ndarray:
        y = None
        if getattr(dataset, "_indices", None) is None:
            y = getattr(dataset, "y", None)
        if isinstance(y, torch.Tensor) and y.numel() == len(dataset):
            return self.check_labels(y.view(-1).cpu().numpy())
        return self.check_labels([int(d.y.view(-1)[0]) for d in dataset])

    @staticmethod
    def kind(method: str) -> Literal["pyg", "hyper", "spectral"]:
        if method == "T-MPHN":
            return "hyper"
        if method in ("T-Spectral", "T-Spatial"):
            return "spectral"
        if method == "GIN" or method in NG_MODELS:
            return "pyg"
        raise ValueError(f"unknown method '{method}', choose from {GRAPH_METHODS}")

    def hyperedges(self, data) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        cfg = self.params.tmphn_config
        x = data.x.detach().cpu().numpy()
        ei = data.edge_index.detach().cpu().numpy()
        n = x.shape[0]
        if ei.size and ei.max() >= n:
            raise ValueError(
                f"edge_index refers to node {ei.max()} but x has {n} rows "
                f"(x shape {x.shape}); x and edge_index come from different "
                "preprocessing (stale dataset cache?)"
            )
        needs_scores = cfg.hyperedges == "knn" or cfg.strategy != "random"
        sim = node_scores(x, getattr(cfg, "scores", "auto")) if needs_scores else None
        if cfg.hyperedges == "edges":
            pairs = np.sort(ei.T, axis=1)
            edges = list(np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0))
        elif cfg.hyperedges == "neighborhood":
            A = np.zeros((n, n), dtype=bool)
            A[ei[0], ei[1]] = True
            A[ei[1], ei[0]] = True
            edges = hyperedges_from_adjacency(A)
        elif cfg.hyperedges == "knn":
            edges = knn_hyperedges(sim, getattr(cfg, "knn_k", 4))
        else:
            raise ValueError(f"unknown hyperedges mode '{cfg.hyperedges}'")
        return x, edges, sim

    def structure_key(self, kind: str) -> tuple:
        cfg = self.params.tmphn_config
        keep = getattr(cfg, "keep_on_device", False)
        return (
            kind,
            tuple(self.mlst()) if kind == "hyper" else None,
            cfg.hyperedges,
            cfg.strategy,
            getattr(cfg, "knn_k", 4),
            getattr(cfg, "scores", "auto"),
            getattr(cfg, "temperature", 1.0),
            self.params.self_loop,
            self.params.seed if cfg.strategy != "topk" else None,
            str(self.device) if keep else "cpu",
        )

    def convert(self, dataset, kind: str) -> List:
        """PyG graphs -> hypergraphs, cached while the structure does not change."""
        key = (id(dataset),) + self.structure_key(kind)
        if key in self._cache and self._cache[key][0] is dataset:
            return self._cache[key][1]
        cfg = self.params.tmphn_config
        build = build_hypergraph if kind == "hyper" else build_spectral_graph
        extra = (self.mlst(),) if kind == "hyper" else ()
        graphs = []
        for i, data in enumerate(
            tqdm(
                dataset,
                disable=not self.progress_bar,
                desc=f"{self.dataset_name}|to_{kind}",
            )
        ):
            x, edges, sim = self.hyperedges(data)
            graph = build(
                x,
                edges,
                *extra,
                y=int(data.y.view(-1)[0]),
                self_loops=self.params.self_loop,
                strategy=cfg.strategy,
                scores=sim,
                temperature=getattr(cfg, "temperature", 1.0),
                seed=self.params.seed + i,
            )
            graphs.append(graph.to(self.device) if key[-1] != "cpu" else graph)
        self._cache = {key: (dataset, graphs)}
        return graphs

    def loaders(self, dataset, method: str, split: TTV) -> Tuple[TTV, int]:
        kind = self.kind(method)
        bs, seed = self.params.batch_size, self.params.seed
        if kind == "pyg":
            num_nodes = int(dataset[0].num_nodes)

            def make(idx, shuffle):
                return DataLoader(
                    dataset[torch.as_tensor(idx, dtype=torch.long)],
                    batch_size=bs,
                    shuffle=shuffle,
                    drop_last=shuffle and len(idx) % bs == 1,
                    generator=torch.Generator().manual_seed(seed) if shuffle else None,
                )

        else:
            graphs = self.convert(dataset, kind)
            num_nodes = max(g.num_nodes for g in graphs)
            Loader = HyperGraphLoader if kind == "hyper" else SpectralGraphLoader

            def make(idx, shuffle):
                return Loader([graphs[i] for i in idx], bs, shuffle=shuffle, seed=seed)

        self.check_memory(method, num_nodes, bs)
        loaders = TTV(
            train=make(split.train, True),
            val=make(split.val, False),
            test=make(split.test, False),
        )
        return loaders, num_nodes

    def build_model(self, method: str, max_nodes: int) -> torch.nn.Module:
        p = self.params
        if method in TENSOR_METHODS:
            model = self.build_tensor_model(method, "graph", max_nodes)
        elif method == "GIN":
            model = GIN(
                in_f=p.num_features,
                hid_c=p.hid_dim,
                out_f=p.num_classes,
                dropout=p.dropout,
            )
        else:
            model = ResidualGNNs(args=p, model_name=method, GNN=NG_MODELS[method])
        return model.to(self.device)

    def train_epoch(self, model, forward, loader, optimizer, weight=None):
        model.train()
        ys, outs, total, count = [], [], 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            y = batch.y.view(-1).long()
            optimizer.zero_grad()
            out = forward(batch)
            loss = F.nll_loss(out, y, weight=weight)
            if not self.step(model, optimizer, loss):
                continue
            ys.append(y.cpu())
            outs.append(out.detach().cpu())
            total += loss.item() * len(y)
            count += len(y)
        if not count:
            raise RuntimeError("every training batch produced a non-finite loss")
        return torch.cat(ys).numpy(), torch.cat(outs).numpy(), total / count

    @torch.no_grad()
    def predict(self, model, forward, loader):
        model.eval()
        ys, outs = [], []
        for batch in loader:
            batch = batch.to(self.device)
            outs.append(forward(batch).cpu())
            ys.append(batch.y.view(-1).cpu())
        return torch.cat(ys).numpy(), torch.cat(outs).numpy(), None

    def train(
        self,
        dataset,
        method: Optional[str] = None,
        split: Optional[TTV] = None,
    ) -> RunResult:
        """Train one method on one split and score every epoch."""
        method = method or self.params.method
        labels = self.labels(dataset)
        self.params.num_features = int(dataset[0].x.shape[1])
        split = split or self.split(labels)
        loaders, num_nodes = self.loaders(dataset, method, split)
        set_seed(self.params.seed)
        model = self.build_model(method, num_nodes)
        forward = self.forward_fn(model)
        weight = self.loss_weights(labels[split.train])

        def train_step(model, optimizer):
            return self.train_epoch(model, forward, loaders.train, optimizer, weight)

        def evaluate(model, train_out):
            return TTV(
                train=train_out,
                val=self.predict(model, forward, loaders.val),
                test=self.predict(model, forward, loaders.test),
            )

        return self.fit(model, method, train_step, evaluate)
