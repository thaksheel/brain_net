import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import (
    DENSE_METHODS,
    TENSOR_METHODS,
    BaseTrainer,
    node_scores,
    set_seed,
)
from .config import TTV
from .scoring import RunResult
from .thgnn import (
    HyperGraph,
    build_hypergraph,
    build_spectral_graph,
    collate,
    hyperedges_from_incidence,
)

NODE_METHODS = TENSOR_METHODS + ("MLP",)


@dataclass
class NodeData:
    X: np.ndarray
    H: Any
    Y: np.ndarray
    name: Optional[str] = None


class FeatureMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        dims = [in_dim] + [hid_dim] * (num_layers - 1)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], num_classes))

    def forward(self, graph) -> torch.Tensor:
        return F.log_softmax(self.net(graph.x), dim=-1)


class NodeTrainer(BaseTrainer):
    """Transductive node classification on one hypergraph, methods in NODE_METHODS."""

    def load_dataset(self, folder: str, name: Optional[str] = None) -> NodeData:
        folder = Path(folder)
        with open(folder / "hypergraph.pickle", "rb") as f:
            H = pickle.load(f)
        with open(folder / "feature.pickle", "rb") as f:
            X = pickle.load(f)
        with open(folder / "labels.pickle", "rb") as f:
            Y = pickle.load(f)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 2:
            Y = Y.argmax(axis=1)
        data = NodeData(
            X=X.astype(np.float32),
            H=H,
            Y=Y.astype(np.int64).reshape(-1),
            name=name or folder.name,
        )
        self.params.dataset = data.name
        self.params.num_features = data.X.shape[1]
        self.params.num_classes = int(data.Y.max()) + 1
        return data

    def labels(self, data: NodeData) -> np.ndarray:
        return self.check_labels(data.Y)

    def split(self, labels: np.ndarray, seed: Optional[int] = None) -> TTV:
        """Paper protocol: the same number of training nodes per class."""
        rng = np.random.default_rng(self.params.seed if seed is None else seed)
        num_classes = int(labels.max()) + 1
        per_class = int(self.params.train_ratio * len(labels) / num_classes)
        train, rest = [], []
        for c in range(num_classes):
            idx = rng.permutation(np.flatnonzero(labels == c))
            train.append(idx[:per_class])
            rest.append(idx[per_class:])
        rest = rng.permutation(np.concatenate(rest))
        n_val = int(self.params.valid_ratio * len(labels))
        return TTV(train=np.concatenate(train), val=rest[:n_val], test=rest[n_val:])

    def convert(self, data: NodeData, method: str):
        cfg = self.params.tmphn_config
        if method == "T-MPHN":
            kind = "hyper"
        elif method in DENSE_METHODS:
            kind = "spectral"
        else:
            kind = "features"
        key = (
            id(data),
            kind,
            tuple(self.mlst()) if kind == "hyper" else None,
            cfg.strategy,
            getattr(cfg, "scores", "auto"),
            getattr(cfg, "temperature", 1.0),
            self.params.self_loop,
            self.params.seed if cfg.strategy != "topk" else None,
        )
        if key in self._cache and self._cache[key][0] is data:
            return self._cache[key][1]
        if kind == "features":
            graph = HyperGraph(
                torch.as_tensor(data.X, dtype=torch.float32),
                [],
                torch.as_tensor(data.Y),
            )
        else:
            edges = hyperedges_from_incidence(data.H)
            scores = None
            if cfg.strategy != "random":
                scores = node_scores(data.X, getattr(cfg, "scores", "auto"))
            kwargs = dict(
                y=data.Y,
                self_loops=self.params.self_loop,
                strategy=cfg.strategy,
                scores=scores,
                temperature=getattr(cfg, "temperature", 1.0),
                seed=self.params.seed,
            )
            if kind == "hyper":
                graph = build_hypergraph(data.X, edges, self.mlst(), **kwargs)
            else:
                graph = build_spectral_graph(data.X, edges, **kwargs)
        self._cache = {key: (data, graph)}
        return graph

    def build_model(self, method: str, num_nodes: int) -> nn.Module:
        p = self.params
        if method in TENSOR_METHODS:
            model = self.build_tensor_model(method, "node", num_nodes)
        elif method == "MLP":
            layers = max(p.num_layers, 2)
            model = FeatureMLP(
                p.num_features, p.hid_dim, p.num_classes, layers, p.dropout
            )
        else:
            raise ValueError(f"unknown method '{method}', choose from {NODE_METHODS}")
        return model.to(self.device)

    def train(
        self,
        data: NodeData,
        method: Optional[str] = None,
        split: Optional[TTV] = None,
    ) -> RunResult:
        """Full-batch training on one split and score every epoch."""
        method = method or self.params.method
        if method not in NODE_METHODS:
            raise ValueError(f"unknown method '{method}', choose from {NODE_METHODS}")
        labels = self.labels(data)
        self.params.num_features = int(data.X.shape[1])
        split = split or self.split(labels)
        graph = self.convert(data, method)
        inputs = (collate([graph]) if method == "T-MPHN" else graph).to(self.device)
        self.check_memory(method, graph.num_nodes, 1)
        set_seed(self.params.seed)
        model = self.build_model(method, graph.num_nodes)
        forward = self.forward_fn(model)
        y = torch.as_tensor(labels, device=self.device)
        train_idx = torch.as_tensor(split.train, dtype=torch.long, device=self.device)
        weight = self.loss_weights(labels[split.train])

        def train_step(model, optimizer):
            model.train()
            optimizer.zero_grad()
            out = forward(inputs)
            loss = F.nll_loss(out[train_idx], y[train_idx], weight=weight)
            self.step(model, optimizer, loss)

        def evaluate(model, train_out):
            model.eval()
            with torch.no_grad():
                out = forward(inputs).cpu().numpy()
            return TTV(
                train=(labels[split.train], out[split.train], None),
                val=(labels[split.val], out[split.val], None),
                test=(labels[split.test], out[split.test], None),
            )

        return self.fit(model, method, train_step, evaluate)
