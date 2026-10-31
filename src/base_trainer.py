import copy
import inspect
import random
import time
import warnings
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import TTV, Params, TMPHNConfig
from .scoring import EpochResult, RunResult, Scoring
from .thgnn import TMPHN, TSpatialHGNN, TSpectralHGNN

TENSOR_METHODS = ("T-MPHN", "T-Spectral", "T-Spatial")
DENSE_METHODS = ("T-Spectral", "T-Spatial")
EXTRA_CONFIG_KEYS = ("readout", "spatial_combine", "temperature")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def node_scores(x: np.ndarray, mode: str = "auto") -> np.ndarray:
    if mode == "features" or (mode == "auto" and x.shape[0] == x.shape[1]):
        return np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.abs(np.corrcoef(x)))


def dense_memory_gb(
    method: str, num_nodes: int, batch_size: int, hid_dim: int, num_layers: int
) -> float:
    n, b = num_nodes, batch_size
    structure = b * n * n * (n + 1) if method == "T-Spectral" else b * n**3
    activations = 3 * b * (n + 1) * n * hid_dim * num_layers
    return 4 * (structure + activations) / 1e9


def params_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: params_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): params_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [params_to_dict(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def _build(cls, values: Dict):
    names = {f.name for f in fields(cls) if f.init}
    obj = cls(**{k: v for k, v in values.items() if k in names})
    for k, v in values.items():
        if k not in names:
            setattr(obj, k, v)
    return obj


def params_from_dict(values: Dict) -> Params:
    values = dict(values)
    config = values.pop("tmphn_config", None)
    params = _build(Params, values)
    if config is not None:
        params.tmphn_config = _build(TMPHNConfig, config)
    return params


def set_param(params: Params, key: str, value: Any):
    *parents, name = key.split(".")
    target = params
    for p in parents:
        target = getattr(target, p)
    if not hasattr(target, name) and name not in EXTRA_CONFIG_KEYS:
        raise AttributeError(f"unknown parameter '{key}'")
    setattr(target, name, value)


def apply_params(params: Params, overrides: Dict[str, Any]) -> Params:
    params = copy.deepcopy(params)
    for k, v in overrides.items():
        set_param(params, k, v)
    return params


class BaseTrainer:
    def __init__(
        self,
        params: Params,
        display: bool = False,
        progress_bar: bool = False,
        monitor: str = "roc_auc",
        patience: Optional[int] = None,
        class_weight: Optional[str] = None,
        clip: Optional[float] = 1.0,
    ):
        self.params = params
        self.display = display
        self.progress_bar = progress_bar
        self.monitor = monitor
        self.patience = patience
        self.class_weight = class_weight
        self.clip = clip
        self.trained_model: Optional[torch.nn.Module] = None
        self._cache: Dict = {}

    @property
    def device(self) -> torch.device:
        return torch.device(self.params.device)

    @property
    def dataset_name(self) -> str:
        return str(self.params.dataset)

    def set_params(self, **params) -> Params:
        for k, v in params.items():
            set_param(self.params, k, v)
        return self.params

    def mlst(self) -> List[int]:
        mlst = list(self.params.Mlst)
        n = self.params.num_layers
        return (mlst + [mlst[-1]] * n)[:n]

    def check_labels(self, labels) -> np.ndarray:
        labels = np.asarray(labels).astype(np.int64).reshape(-1)
        if labels.min() < 0:
            raise ValueError("labels must be integers in 0..num_classes-1")
        self.params.num_classes = int(labels.max()) + 1
        return labels

    def split(self, labels: np.ndarray, seed: Optional[int] = None) -> TTV:
        seed = self.params.seed if seed is None else seed
        train_ratio, valid_ratio = self.params.train_ratio, self.params.valid_ratio
        rest, test = train_test_split(
            np.arange(len(labels)),
            test_size=round(1 - train_ratio - valid_ratio, 10),
            stratify=labels,
            random_state=seed,
        )
        train, val = train_test_split(
            rest,
            test_size=round(valid_ratio / (train_ratio + valid_ratio), 10),
            stratify=labels[rest],
            random_state=seed,
        )
        return TTV(train=train, val=val, test=test)

    def build_tensor_model(
        self, method: str, task: str, max_nodes: Optional[int] = None
    ) -> torch.nn.Module:
        p, cfg = self.params, self.params.tmphn_config
        common = dict(
            in_dim=p.num_features,
            hid_dim=p.hid_dim,
            num_classes=p.num_classes,
            dropout=p.dropout,
            norm=cfg.norm,
            task=task,
            pool=cfg.pool,
            project_input=cfg.project_input,
        )
        if method == "T-MPHN":
            return TMPHN(
                Mlst=self.mlst(),
                combine=p.combine,
                chunk_size=cfg.chunk_size,
                checkpoint=cfg.checkpoint,
                **common,
            )
        if method == "T-Spectral":
            return TSpectralHGNN(
                num_layers=p.num_layers,
                max_nodes=max_nodes,
                combine=p.combine,
                readout=getattr(cfg, "readout", "learned"),
                **common,
            )
        if method == "T-Spatial":
            return TSpatialHGNN(
                num_layers=p.num_layers,
                combine=getattr(cfg, "spatial_combine", "sum"),
                **common,
            )
        raise ValueError(f"'{method}' is not a tensor method {TENSOR_METHODS}")

    def forward_fn(self, model: torch.nn.Module) -> Callable:
        if "terms" in inspect.signature(model.forward).parameters:
            return lambda b: F.log_softmax(
                model(b.x, b.terms, b.batch, b.num_graphs), dim=-1
            )
        return lambda b: F.log_softmax(model(b), dim=-1)

    def loss_weights(self, labels: np.ndarray) -> Optional[torch.Tensor]:
        if self.class_weight is None:
            return None
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")
        counts = np.bincount(labels, minlength=self.params.num_classes)
        weights = len(labels) / (self.params.num_classes * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def check_memory(self, method: str, num_nodes: int, batch_size: int):
        if method not in DENSE_METHODS or self.device.type != "cuda":
            return
        need = dense_memory_gb(
            method, num_nodes, batch_size, self.params.hid_dim, self.params.num_layers
        )
        free = torch.cuda.mem_get_info(self.device.index)[0] / 1e9
        if need > free:
            warnings.warn(
                f"{method} needs ~{need:.1f} GB ({num_nodes} nodes, "
                f"batch_size={batch_size}) but {free:.1f} GB is free; "
                "lower batch_size or hid_dim, or use T-MPHN"
            )

    def step(self, model: torch.nn.Module, optimizer, loss: torch.Tensor) -> bool:
        if not torch.isfinite(loss):
            warnings.warn("non-finite loss, optimizer step skipped")
            return False
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        optimizer.step()
        return True

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device.index)

    def log(self, r: EpochResult):
        m = self.monitor
        print(
            f"{r.dataset} {r.method} seed={r.seed} epoch={r.epoch} "
            f"loss={r.loss.train:.4f} val_{m}={r.metric(m, 'val'):.4f} "
            f"test_{m}={r.metric(m, 'test'):.4f} test_acc={r.accuracy.test:.4f} "
            f"{r.duration:.2f}s"
        )

    def fit(
        self,
        model: torch.nn.Module,
        method: str,
        train_step: Callable,
        evaluate: Callable,
    ) -> RunResult:
        """Shared epoch loop: scores every epoch and keeps the best validation epoch."""
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.params.lr, weight_decay=self.params.wd
        )
        scoring = Scoring(self.params.num_classes, self.monitor)
        epochs: List[EpochResult] = []
        best, best_test, best_state, stale = None, None, None, 0
        start = time.perf_counter()
        for epoch in tqdm(
            range(1, self.params.epochs + 1),
            disable=not self.progress_bar,
            desc=f"{self.dataset_name}|{method}|seed={self.params.seed}",
        ):
            self.sync()
            t0 = time.perf_counter()
            train_out = train_step(model, optimizer)
            self.sync()
            duration = time.perf_counter() - t0
            outputs = evaluate(model, train_out)
            result = scoring.score_epoch(
                epoch, method, self.dataset_name, self.params.seed, duration, outputs
            )
            epochs.append(result)
            if scoring.is_better(result, best):
                best, best_test, stale = result, outputs.test, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                stale += 1
            if self.display:
                self.log(result)
            if self.patience and stale >= self.patience:
                break
        if best is None:
            raise ValueError("params.epochs must be at least 1")
        model.load_state_dict(best_state)
        self.trained_model = model
        return RunResult(
            dataset=self.dataset_name,
            method=method,
            seed=self.params.seed,
            epochs=epochs,
            best=best,
            y_true=np.asarray(best_test[0]),
            y_prob=scoring.probabilities(best_test[1]),
            run_time=time.perf_counter() - start,
            params=params_to_dict(self.params),
        )
