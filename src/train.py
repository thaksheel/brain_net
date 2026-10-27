import itertools, torch, time, random
import pandas as pd
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.nn import (
    APPNP,
    MLP,
    GCNConv,
    GINConv,
    SAGEConv,
    GraphConv,
    TransformerConv,
    ChebConv,
    GATConv,
    SGConv,
    GeneralConv,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Dict, List, Optional, Literal, Tuple
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .utils import (
    NeighborFinder,
    NeuroGraphDataset,
    initialize,
    read_data,
    add_self_loop,
)
from .models import GIN, ResidualGNNs
from . import (
    Params,
    EvalResults,
    TTV,
    GridSearchParams,
    TMPHN,
    knn_hyperedges,
    hyperedges_from_adjacency,
    build_hypergraph,
    HyperGraphLoader,
    HyperGraph,
)
from .tmphn.tspectral import build_spectral_graph, TSpectralHGNN, SpectralGraphLoader, SpectralGraph
from .tmphn.tspatial import TSpatialHGNN


class GraphTrainer:
    def __init__(
        self,
        params: Params,
        display: bool = False,
        progress_bar: bool = False,
    ):
        self.params = params
        self.display = display
        self.progress_bar = progress_bar
        self.device = torch.device(self.params.device)

        # Null
        self.data: Dict = None
        self.trained_model: torch.nn.Module = None
        self.durations: List[float] = [time.time()]

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.params, k, v)
        return self.params

    def initialize_node(
        self,
        X: np.ndarray,
        H: np.ndarray,
        Y: np.ndarray,
        datafile: str = None,
    ):
        if datafile is not None:
            H, X, Y = read_data(datafile)
        self.params.num_classes = len(np.unique(Y))
        if self.display:
            print("====> initializing model, optimizer, and data splits")
        model, optimizer, train_idx, val_idx, test_idx, data = initialize(
            H,
            X,
            Y,
            self.params,
            seed_split=self.params.seed_splits,
            device=self.device,
        )
        torch.manual_seed(self.params.seed_weights)
        np.random.seed(self.params.seed_weights)
        random.seed(self.params.seed_weights)
        model.reset_parameters()
        idx = TTV(train=train_idx, test=test_idx, val=val_idx)
        self.data = data
        return model, optimizer, data, idx

    def initialize_graph(self, Hs, Xs, Ys) -> List[Data]:
        dataset = []
        for H, X, Y in zip(Hs, Xs, Ys):
            if self.params.self_loop:
                H = add_self_loop(H)
            X = torch.from_numpy(X).float().to(self.device)
            Y = torch.from_numpy(np.array(Y)).long().to(self.device)
            neigh_dict = NeighborFinder(H, self.params.M).neig_for_targets(
                list(range(X.shape[0]))
            )
            H = torch.from_numpy(H).long().to(self.device)
            dataset.append(Data(x=X, adj=H, neig_dict=neigh_dict, y=Y))
        return dataset

    def graph_edges(self, edge_index: np.ndarray) -> List[np.ndarray]:
        """Unique undirected edges {u, v} (u != v) as size-2 hyperedges."""
        pairs = np.sort(edge_index.T, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        return list(np.unique(pairs, axis=0))

    def node_scores(self, x: np.ndarray, mode: str) -> np.ndarray:
        if mode == "features" or (mode == "auto" and x.shape[0] == x.shape[1]):
            return np.abs(x)  # node features are FC rows -> |FC|
        return np.nan_to_num(np.abs(np.corrcoef(x)))

    def to_hypergraphs(self, dataset: InMemoryDataset) -> List[HyperGraph]:
        needs_scores = (
            self.params.tmphn_config.hyperedges == "knn"
            or self.params.tmphn_config.strategy != "random"
        )
        graphs: List[HyperGraph] = []
        start = time.perf_counter()
        for i, data in enumerate(dataset):
            x = data.x.numpy()
            n = x.shape[0]
            ei = data.edge_index.numpy()
            sim = (
                self.node_scores(x, self.params.tmphn_config.scores)
                if needs_scores
                else None
            )
            if self.params.tmphn_config.hyperedges == "edges":
                edges = self.graph_edges(ei)
            elif self.params.tmphn_config.hyperedges == "neighborhood":
                A = np.zeros((n, n), dtype=bool)
                A[ei[0], ei[1]] = True
                A[ei[1], ei[0]] = True
                edges = hyperedges_from_adjacency(A)
            elif self.params.tmphn_config.hyperedges == "knn":
                edges = knn_hyperedges(sim, self.params.tmphn_config.knn_k)
            else:
                raise ValueError(
                    f"unknown hyperedges mode '{self.params.tmphn_config.hyperedges}'"
                )
            graphs.append(
                build_hypergraph(
                    x,
                    edges,
                    self.params.Mlst,
                    y=int(data.y.view(-1)[0]),
                    self_loops=self.params.self_loop,
                    strategy=self.params.tmphn_config.strategy,
                    scores=sim,
                    seed=self.params.seed + i,
                )
            )
            if self.params.tmphn_config.log_every and (i + 1) % 500 == 0:
                rate = (time.perf_counter() - start) / (i + 1)
                print(
                    f"  converted {i + 1}/{len(dataset)} subjects "
                    f"(~{rate * (len(dataset) - i - 1):.0f}s left)"
                )
        return graphs

    def load_dataset(
        self,
        root_folder: str,
        dataset_name: str,
    ) -> NeuroGraphDataset:
        dataset = NeuroGraphDataset(root=root_folder, name=dataset_name)
        self.params.num_features = dataset.num_features
        self.params.num_classes = dataset.num_classes
        return dataset

    def subset_dataset(
        self, dataset: InMemoryDataset, indices: list[int]
    ) -> InMemoryDataset:
        data_list = [dataset[i] for i in indices]
        data, slices = dataset.collate(data_list)
        new_ds = dataset.__class__(root=dataset.root, name=dataset.name)
        new_ds.data, new_ds.slices = data, slices
        return new_ds

    def ttv_splits(self, dataset: InMemoryDataset) -> TTV:
        labels = [d.y.item() for d in dataset]
        c = self.params.train_ratio + self.params.valid_ratio
        train_tmp, test_indices = train_test_split(
            list(range(len(labels))),
            test_size=1 - c,
            stratify=labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        tmp = [dataset[i] for i in train_tmp]
        train_labels = [d.y.item() for d in tmp]
        train_indices, val_indices = train_test_split(
            list(range(len(train_labels))),
            test_size=self.params.valid_ratio,
            stratify=train_labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        # Convert back to InMemoryDataset
        train_dataset = self.subset_dataset(
            dataset, [train_tmp[i] for i in train_indices]
        )
        val_dataset = self.subset_dataset(dataset, [train_tmp[i] for i in val_indices])
        test_dataset = self.subset_dataset(dataset, test_indices)
        return TTV(train=train_dataset, test=test_dataset, val=val_dataset)

    def to_hypergraph_ttv(self, dataset: InMemoryDataset) -> TTV:
        """PyG graphs -> HyperGraph objects, one list per split. Run once per dataset."""
        labels = [d.y.item() for d in dataset]
        c = self.params.train_ratio + self.params.valid_ratio
        train_tmp, test_indices = train_test_split(
            list(range(len(labels))),
            test_size=1 - c,
            stratify=labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        tmp = [dataset[i] for i in train_tmp]
        train_labels = [d.y.item() for d in tmp]
        train_indices, val_indices = train_test_split(
            list(range(len(train_labels))),
            test_size=self.params.valid_ratio,
            stratify=train_labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        # Convert back to InMemoryDataset
        train_dataset = self.subset_dataset(
            dataset, [train_tmp[i] for i in train_indices]
        )
        val_dataset = self.subset_dataset(dataset, [train_tmp[i] for i in val_indices])
        test_dataset = self.subset_dataset(dataset, test_indices)
        return TTV(
            train=self.to_hypergraphs(train_dataset),
            val=self.to_hypergraphs(val_dataset),
            test=self.to_hypergraphs(test_dataset),
        )


    def get_dataloader(self, ttv_dataset: TTV) -> TTV:
        """REPLACES your get_dataloader: picks the loader that matches the converted graphs."""
        first = ttv_dataset.train[0]
        Loader = (
            SpectralGraphLoader if isinstance(first, SpectralGraph) else HyperGraphLoader
        )
        bs, seed = self.params.batch_size, self.params.seed
        return TTV(
            train=Loader(ttv_dataset.train, bs, shuffle=True, seed=seed),
            val=Loader(ttv_dataset.val, bs, shuffle=False),
            test=Loader(ttv_dataset.test, bs, shuffle=False),
        )

    def compute_alt_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "npv": npv,
        }

    def compute_auc_metrics(self, y_true, logits):
        # Convert logits → probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        roc_auc = roc_auc_score(y_true, probs, multi_class="ovo")
        pr_auc = average_precision_score(y_true, probs)
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    def get_eval_results(
        self,
        epoch: int,
        method: str,
        duration: float,
        train_loss: float,
        test_loss: float,
        val_loss: float,
        train_true: List,
        train_pred: List,
        test_true: List,
        test_pred: List,
        val_true: List,
        val_pred: List,
        train_logits: torch.Tensor,
        test_logits: torch.Tensor,
        val_logits: torch.Tensor,
    ):
        # Medical metrics
        if self.params.num_classes == 2:
            train_med = self.compute_alt_metrics(train_true, train_pred)
            test_med = self.compute_alt_metrics(test_true, test_pred)
            val_med = self.compute_alt_metrics(val_true, val_pred)
            # AUC metrics (requires logits)
            train_auc = self.compute_auc_metrics(train_true, train_logits)
            test_auc = self.compute_auc_metrics(test_true, test_logits)
            val_auc = self.compute_auc_metrics(val_true, val_logits)
        else:
            train_med = {
                "precision": None,
                "recall": None,
                "sensitivity": None,
                "specificity": None,
                "npv": None,
            }
            test_med = {
                "precision": None,
                "recall": None,
                "sensitivity": None,
                "specificity": None,
                "npv": None,
            }
            val_med = {
                "precision": None,
                "recall": None,
                "sensitivity": None,
                "specificity": None,
                "npv": None,
            }
            train_auc = {
                "roc_auc": None,
                "pr_auc": None,
            }
            test_auc = {
                "roc_auc": None,
                "pr_auc": None,
            }
            val_auc = {
                "roc_auc": None,
                "pr_auc": None,
            }

        return EvalResults(
            epoch=epoch,
            method=method,
            duration=duration,
            accuracy=TTV(
                train=accuracy_score(train_true, train_pred),
                test=accuracy_score(test_true, test_pred),
                val=accuracy_score(val_true, val_pred),
            ),
            precision=TTV(
                train=train_med["precision"],
                test=test_med["precision"],
                val=val_med["precision"],
            ),
            recall=TTV(
                train=train_med["recall"],
                test=test_med["recall"],
                val=val_med["recall"],
            ),
            sensitivity=TTV(
                train=train_med["sensitivity"],
                test=test_med["sensitivity"],
                val=val_med["sensitivity"],
            ),
            specificity=TTV(
                train=train_med["specificity"],
                test=test_med["specificity"],
                val=val_med["specificity"],
            ),
            npv=TTV(
                train=train_med["npv"],
                test=test_med["npv"],
                val=val_med["npv"],
            ),
            roc_auc=TTV(
                train=train_auc.get("roc_auc"),
                test=test_auc.get("roc_auc"),
                val=val_auc.get("roc_auc"),
            ),
            pr_auc=TTV(
                train=train_auc.get("pr_auc"),
                test=test_auc.get("pr_auc"),
                val=val_auc.get("pr_auc"),
            ),
            f1=TTV(
                train=f1_score(train_true, train_pred, average=None),
                test=f1_score(test_true, test_pred, average=None),
                val=f1_score(val_true, val_pred, average=None),
            ),
            f1_macro=TTV(
                train=f1_score(train_true, train_pred, average="macro"),
                test=f1_score(test_true, test_pred, average="macro"),
                val=f1_score(val_true, val_pred, average="macro"),
            ),
            loss=TTV(train=train_loss, test=test_loss, val=val_loss),
            rmse=TTV(
                train=root_mean_squared_error(train_true, train_pred),
                test=root_mean_squared_error(test_true, test_pred),
                val=root_mean_squared_error(val_true, val_pred),
            ),
            mae=TTV(
                train=mean_absolute_error(train_true, train_pred),
                test=mean_absolute_error(test_true, test_pred),
                val=mean_absolute_error(val_true, val_pred),
            ),
            M=self.params.M,
            seed=self.params.seed,
            params=self.params,
        )

    def evaluate_graph_cls(
        self,
        dataset: InMemoryDataset,
    ) -> List[EvalResults]:
        """Generate dataloader, batches, optimizer, and criterion to feed into `train()` and `test()` to return `List[EvalResults]`"""
        ttv_dataset = self.to_hypergraph_ttv(dataset)
        ttv_loader = self.get_dataloader(ttv_dataset)
        if self.display:
            print("---> started T-HGNN training...")
        model = TMPHN(
            in_dim=self.params.num_features,
            hid_dim=self.params.hid_dim,
            num_classes=self.params.num_classes,
            Mlst=self.params.Mlst,
            combine=self.params.combine,
            dropout=self.params.dropout,
            norm=self.params.tmphn_config.norm,
            task=self.params.task,
            pool=self.params.tmphn_config.pool,
            chunk_size=self.params.tmphn_config.chunk_size,
            checkpoint=self.params.tmphn_config.checkpoint,
            project_input=self.params.tmphn_config.project_input,
        ).to(self.device)
        eval_results = self.train_graph_cls(model, ttv_loader)
        return eval_results

    def get_base_gnn(self, model_name: str):
        base_models = {
            "APPNP": APPNP,
            "MLP": MLP,
            "GCNConv": GCNConv,
            "GINConv": GINConv,
            "SAGEConv": SAGEConv,
            "GraphConv": GraphConv,
            "TransformerConv": TransformerConv,
            "ChebConv": ChebConv,
            "GATConv": GATConv,
            "SGConv": SGConv,
            "GeneralConv": GeneralConv,
        }
        return base_models[model_name]

    def evaluate_ng_cls(
        self,
        dataset: InMemoryDataset,
        model_name: Literal[
            "GCNConv",
            "GINConv",
            "GraphConv",
            "TransformerConv",
            "GATConv",
            "GeneralConv",
            "SGConv",
            "ChebConv",
            "SAGEConv",
            "APPNP",
            "MLP",
        ] = "GCNConv",
    ):
        ttv_dataset = self.ttv_splits(dataset)
        ttv_loader = self.get_dataloader(ttv_dataset)
        model = ResidualGNNs(
            args=self.params,
            model_name=model_name,
            GNN=self.get_base_gnn(model_name),
        )
        eval_results = self.train_graph_cls(model, ttv_loader)
        return eval_results

    def evaluate_graph_cls_stnd(self, dataset: InMemoryDataset):
        ttv_dataset = self.ttv_splits(dataset)
        ttv_loader = self.get_dataloader(ttv_dataset)
        model = GIN(
            in_f=self.params.num_features,
            hid_c=self.params.hid_dim,
            out_f=self.params.num_classes,
            dropout=self.params.dropout,
        ).to(self.device)
        eval_results = self.train_graph_cls(model, ttv_loader)
        return eval_results

    def run_stratified_n_iteractions(
        self,
        seeds: List[int],
        dataset: InMemoryDataset,
        graph_type: Literal["stnd", "tensor", "ng"],
        model_name: str = None,
    ):
        evals: List[EvalResults] = []
        for seed in seeds:
            self.params.seed = seed
            if graph_type == "stnd":
                eval_results = self.evaluate_graph_cls_stnd(dataset)
                evals.append(eval_results)
            elif graph_type == "tensor":
                eval_results = self.evaluate_graph_cls(dataset)
                evals.append(eval_results)
            elif graph_type == "ng":
                eval_results = self.evaluate_ng_cls(dataset, model_name)
                evals.append(eval_results)
            else:
                raise NotImplementedError(
                    f"---> graph_type={graph_type} not implemented yet."
                )
        return evals

    @torch.no_grad()
    def predict(model, graphs, device, batch_size):
        model.eval()
        outs, ys = [], []
        for b in HyperGraphLoader(graphs, batch_size, shuffle=False):
            b = b.to(device, non_blocking=True)
            outs.append(model(b.x, b.terms, b.batch, b.num_graphs).cpu())
            ys.append(b.y.cpu())
        return torch.cat(outs), torch.cat(ys)

    def optimize_graph_cls(
        self,
        loader: HyperGraphLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
    ):
        """One training pass. Returns (trues, preds, mean_loss, logits)."""
        model.train()  # re-enable dropout every epoch
        out: torch.Tensor
        losses, trues, preds, outs = [], [], [], []
        for i, batch in enumerate(loader):
            # forward
            batch = batch.to(self.device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at batch {i}, skipping step")
                continue

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # bookkeeping
            losses.append(loss.item())
            trues.extend(batch.y.tolist())
            preds.extend(out.argmax(dim=1).tolist())
            outs.append(out.detach())
        return trues, preds, float(np.mean(losses)), torch.cat(outs)

    def train_graph_cls(
        self,
        model: torch.nn.Module,
        ttv_loader: TTV,
    ):
        """Train GNN from loader/batches to return predictates and embeddings"""
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.params.lr, weight_decay=self.params.wd
        )
        results = []
        for epoch in tqdm(
            range(self.params.epochs),
            disable=not self.progress_bar,
            desc=f"graph_train_seed:{self.params.seed}",
        ):
            self._sync(self.params.device)
            start = time.perf_counter()
            trt, trp, tr_loss, out_tr = self.optimize_graph_cls(
                ttv_loader.train, model, optimizer
            )
            self._sync(self.params.device)
            duration = time.perf_counter() - start
            tet, tep, te_loss, out_te = self.model_evaluation(model, ttv_loader.test)
            valt, valp, val_loss, out_val = self.model_evaluation(model, ttv_loader.val)
            results.append(
                self.get_eval_results(
                    epoch + 1,
                    self.params.method,
                    duration,
                    tr_loss,
                    te_loss,
                    val_loss,
                    trt,
                    trp,
                    tet,
                    tep,
                    valt,
                    valp,
                    out_tr,
                    out_te,
                    out_val,
                )
            )
            if self.display:
                print(
                    f"ds_name={self.params.dataset} "
                    f"epoch={epoch+1} "
                    f"loss={tr_loss:.3f} "
                    f"test_acc={results[-1].accuracy.test*100:.2f}% "
                    f"test_f1_macro={results[-1].f1_macro.test*100:.2f}% "
                    f"test_f1={results[-1].f1.test} "
                    f"duration={duration:.2f}s "
                )
        self.trained_model = model
        return results

    @torch.no_grad()
    def model_evaluation(self, model: torch.nn.Module, loader: HyperGraphLoader):
        """Evaluation pass. Returns (trues, preds, mean_loss, logits)."""
        model.eval()
        loss_sum, n = 0.0, 0
        trues, preds, outs = [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)
            loss_sum += F.nll_loss(out, batch.y, reduction="sum").item()
            n += batch.num_graphs
            trues.extend(batch.y.tolist())
            preds.extend(out.argmax(dim=1).tolist())
            outs.append(out)
        return trues, preds, loss_sum / n, torch.cat(outs)

    def _hyperedges_for(self, data):
        """PyG Data -> (x, hyperedges, scores) using tmphn_config, exactly as to_hypergraphs does."""
        cfg = self.params.tmphn_config
        x = data.x.numpy()
        n = x.shape[0]
        ei = data.edge_index.numpy()
        needs_scores = cfg.hyperedges == "knn" or cfg.strategy != "random"
        sim = self.node_scores(x, cfg.scores) if needs_scores else None
        if cfg.hyperedges == "edges":
            edges = self.graph_edges(ei)
        elif cfg.hyperedges == "neighborhood":
            A = np.zeros((n, n), dtype=bool)
            A[ei[0], ei[1]] = True
            A[ei[1], ei[0]] = True
            edges = hyperedges_from_adjacency(A)
        elif cfg.hyperedges == "knn":
            edges = knn_hyperedges(sim, cfg.knn_k)
        else:
            raise ValueError(f"unknown hyperedges mode '{cfg.hyperedges}'")
        return x, edges, sim

    def to_spectral_graphs(self, dataset):
        cfg = self.params.tmphn_config
        graphs = []
        for i, data in enumerate(dataset):
            x, edges, sim = self._hyperedges_for(data)
            graphs.append(
                build_spectral_graph(
                    x,
                    edges,
                    y=int(data.y.view(-1)[0]),
                    self_loops=self.params.self_loop,
                    strategy=cfg.strategy,
                    scores=sim,
                    seed=self.params.seed + i,
                )
            )
        return graphs

    def to_spectral_ttv(self, dataset) -> "TTV":
        """Same stratified split as ttv_splits, then convert each split once."""
        ttv = self.ttv_splits(dataset)
        return TTV(
            train=self.to_spectral_graphs(ttv.train),
            val=self.to_spectral_graphs(ttv.val),
            test=self.to_spectral_graphs(ttv.test),
        )

    def build_tspectral(self, max_nodes: int, task: str) -> torch.nn.Module:
        cfg = self.params.tmphn_config
        return TSpectralHGNN(
            in_dim=self.params.num_features,
            hid_dim=self.params.hid_dim,
            num_classes=self.params.num_classes,
            num_layers=self.params.num_layers,
            max_nodes=max_nodes,
            combine=self.params.combine,
            dropout=self.params.dropout,
            norm=cfg.norm,
            readout=getattr(cfg, "readout", "learned"),
            task=task,
            pool=cfg.pool,
            project_input=cfg.project_input,
        ).to(self.device)

    def build_tspatial(self, task: str) -> torch.nn.Module:
        cfg = self.params.tmphn_config
        return TSpatialHGNN(
            in_dim=self.params.num_features,
            hid_dim=self.params.hid_dim,
            num_classes=self.params.num_classes,
            num_layers=self.params.num_layers,
            combine=getattr(
                cfg, "spatial_combine", "sum"
            ),  # "sum" (paper, with self-loops) | "concat" | "none"
            dropout=self.params.dropout,
            norm=cfg.norm,
            task=task,
            pool=cfg.pool,
            project_input=cfg.project_input,
        ).to(self.device)

    def evaluate_graph_cls_spectral(self, dataset):
        """T-spectral counterpart of evaluate_graph_cls (TMPHN); returns List[EvalResults]."""
        ttv_dataset = self.to_spectral_ttv(dataset)
        ttv_loader = self.get_dataloader(ttv_dataset)
        max_nodes = max(
            g.num_nodes
            for split in (ttv_dataset.train, ttv_dataset.val, ttv_dataset.test)
            for g in split
        )
        torch.manual_seed(self.params.seed)
        model = self.build_tspectral(max_nodes, task="graph")
        return self.train_graph_cls(model, ttv_loader) 

    def evaluate_graph_cls_spatial(self, dataset):
        """T-spatial counterpart of evaluate_graph_cls (TMPHN); returns List[EvalResults]."""
        ttv_dataset = self.to_spectral_ttv(dataset)  # same data format as T-spectral
        ttv_loader = self.get_dataloader(ttv_dataset)
        torch.manual_seed(self.params.seed)
        model = self.build_tspatial(task="graph")
        return self.train_graph_cls(model, ttv_loader)

    def train_node_cls(
        self,
        X: np.ndarray,
        H: np.ndarray,
        Y: np.ndarray,
        datafile: str = None,
    ):
        model, optimizer, data, idx = self.initialize_node(X, H, Y, datafile=datafile)
        if self.params.method == "T-Spectral" or self.params.method == "T-Spatial":
            A, X = data["hypergraph"], data["X"]  # adjacency tensor
            A = torch.from_numpy(A).to(self.device)
        X = torch.from_numpy(X).to(self.device)
        Y = data["Y"]
        Y: torch.Tensor = Y.to(self.device)
        train_results: list[EvalResults] = []
        for epoch in range(self.params.epochs):
            start_epoch = time.time()
            model.train()
            optimizer.zero_grad()
            if self.params.method == "T-Spectral" or self.params.method == "T-Spatial":
                output_train = model.forward(A, X)[idx.train]
            elif self.params.method == "T-MPHN":
                output_train = model.forward(idx.train)
            train_loss = F.nll_loss(output_train, Y[idx.train])
            train_loss.backward()
            optimizer.step()
            train_results.append(
                self.node_cls_evaluation(
                    model=model,
                    data=data,
                    idx=idx,
                    epoch=epoch,
                    duration=time.time() - start_epoch,
                    method=self.params.method,
                    M=self.params.M,
                )
            )
        return train_results

    def _sync(self, device):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()  # so timings include the GPU work

    def node_cls_evaluation(
        self, model: torch.nn.Module, data: Dict, idx: TTV, **kwargs
    ):
        Y = data["Y"]
        was_training = model.training
        model.eval()
        with torch.no_grad():
            if self.params.method in ("T-Spectral", "T-Spatial"):
                A, X = data["hypergraph"], data["X"]
                output = model(A, X)  # single transductive forward pass
                output_train = output[idx.train]
                output_val = output[idx.val]
                output_test = output[idx.test]
            elif self.params.method == "T-MPHN":
                output_train = model(idx.train)
                output_val = model(idx.val)
                output_test = model(idx.test)
            else:
                raise ValueError(f"Unknown method: {self.params.method}")
            tr_loss = F.nll_loss(output_train, Y[idx.train])
            val_loss = F.nll_loss(output_val, Y[idx.val])
            te_loss = F.nll_loss(output_test, Y[idx.test])
        if was_training:
            model.train()
        prediction_train = output_train.argmax(1).cpu().numpy()
        prediction_val = output_val.argmax(1).cpu().numpy()
        prediction_test = output_test.argmax(1).cpu().numpy()
        tr = self.get_eval_results(
            kwargs["epoch"],
            kwargs["method"],
            kwargs["duration"],
            tr_loss.cpu().numpy(),
            te_loss.cpu().numpy(),
            val_loss.cpu().numpy(),
            Y[idx.train].cpu().numpy(),
            prediction_train,
            Y[idx.test].cpu().numpy(),
            prediction_test,
            Y[idx.val].cpu().numpy(),
            prediction_val,
            train_logits=output_train.cpu(),
            test_logits=output_test.cpu(),
            val_logits=output_val.cpu(),
        )
        if self.display:
            print(
                f"epoch={tr.epoch} "
                f"loss={tr.loss.train:.4f} "
                f"test_acc={tr.accuracy.test*100:.2f}% "
                f"f1_test={tr.f1_macro.test*100:.2f}% "
                f"duration={tr.duration:.3f}s "
                f"method={tr.method}"
            )
        return tr

    def get_best_scores(self, train_results: List[EvalResults]) -> EvalResults:
        """Find best training results using highest `val` accuracy"""
        test_score = np.array([tr.accuracy.test for tr in train_results])
        best = np.argmax(test_score)
        return train_results[best]

    def quick_plot(self, epochs, data, **kwargs):
        plt.plot(epochs, data, marker="o", color="steelblue")
        plt.xlabel("epochs")
        plt.ylabel("data")
        if "model" in kwargs:
            plt.title(kwargs["model"])
        plt.show()

    def plot_adjacency(self, A: np.ndarray):
        plt.figure(figsize=(10, 8))
        sns.heatmap(A, cmap="grey", cbar=True)
        plt.title("adjacency matrix")
        plt.show()

    def exhaustive_search(self, param_grid: Dict, datafile: str = None):
        # TODO: need to tune on validation instead of full dataset!
        keys, values = zip(*param_grid.items())
        best_score, best_params = 0, None
        result_arr = []
        param_arr = []
        combos = list(itertools.product(*values))
        for k, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            self.set_params(**params)
            train_results = self.fit_train(datafile=datafile)
            best_result = self.get_best_scores(train_results)
            result_arr.append(best_result)
            param_arr.append(params)
            if best_result.accuracy.test > best_score:
                best_score, best_params = best_result.accuracy.test, params
            if self.display:
                print(
                    f"---> params eval, best_results={best_score} best_params={best_params} progress={k+1}/{len(combos)}"
                )
        return np.array(result_arr), np.array(param_arr)

    def grid_search(
        self,
        dataset: InMemoryDataset,
        param_grid: GridSearchParams,
        graph_type: Literal["stnd", "tensor", "ng"],
        model_name: str = None,
        maxiter: int = 50,
    ):
        best_score, best_params = 0, None
        evals: Dict[int, EvalResults] = {}
        for i in tqdm(range(maxiter), desc="Grid Search"):
            params = {k: random.choice(v) for k, v in param_grid.__dict__.items()}
            self.set_params(**params)
            if graph_type == "stnd":
                evals[i] = self.evaluate_graph_cls_stnd(dataset)
            elif graph_type == "tensor":
                evals[i] = self.evaluate_graph_cls(dataset)
            elif graph_type == "ng":
                evals[i] = self.evaluate_ng_cls(dataset, model_name=model_name)
            best_result = self.get_best_scores(evals[i])
            # TODO: what is the right selection mechansim? test? train? or val?
            if best_result.accuracy.val > best_score:
                best_score, best_params = best_result.accuracy.val, params
        return best_params, best_score

    def evaluation_results_to_df(
        self,
        evaluation_results: List[EvalResults],
        outname: str,
        export: bool = False,
        exclude_fields: List[str] = ["params"],
    ):
        data = []
        for er in evaluation_results:
            d = er.__dict__
            result = {}
            for k, v in d.items():
                if isinstance(v, TTV):
                    result[k + "_train"] = (
                        v.train.item() if isinstance(v.train, torch.Tensor) else v.train
                    )
                    result[k + "_test"] = (
                        v.test.item() if isinstance(v.test, torch.Tensor) else v.test
                    )
                    result[k + "_val"] = (
                        v.val.item() if isinstance(v.val, torch.Tensor) else v.val
                    )
                    exclude_fields.append(k)
                result[k] = v
            data.append({k: v for k, v in result.items() if k not in exclude_fields})
        df = pd.DataFrame(data)
        if export:
            df.to_excel(outname)
        return df

    def get_best_eval_results(self, eval_results: List[EvalResults]) -> EvalResults:
        test_score = np.array([er.accuracy.val for er in eval_results])
        best = np.argmax(test_score)
        return eval_results[best]
