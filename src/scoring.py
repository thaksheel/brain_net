import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

from .config import TTV

METRICS = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "mcc",
)
BINARY_ONLY = ("pr_auc", "sensitivity", "specificity", "npv")
LOWER_IS_BETTER = ("loss",)
SPLITS = ("train", "val", "test")


@dataclass
class EpochResult:
    dataset: str
    method: str
    seed: int
    epoch: int
    duration: float
    loss: TTV
    accuracy: TTV
    balanced_accuracy: TTV
    roc_auc: TTV
    pr_auc: TTV
    sensitivity: TTV
    specificity: TTV
    precision: TTV
    npv: TTV
    f1: TTV
    mcc: TTV

    def metric(self, name: str, split: str = "val") -> float:
        return getattr(getattr(self, name), split)

    def to_dict(self) -> Dict:
        row = {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "epoch": self.epoch,
            "duration": self.duration,
        }
        for m in METRICS:
            for s in SPLITS:
                row[f"{m}_{s}"] = self.metric(m, s)
        return row

    def __repr__(self) -> str:
        lines = [
            f"EpochResult(dataset={self.dataset}, method={self.method}, "
            f"seed={self.seed}, epoch={self.epoch}, duration={self.duration:.3f}s)"
        ]
        for m in METRICS:
            values = "  ".join(f"{s}={self.metric(m, s):.4f}" for s in SPLITS)
            lines.append(f"  {m:<18}{values}")
        return "\n".join(lines)


@dataclass
class RunResult:
    dataset: str
    method: str
    seed: int
    epochs: List[EpochResult]
    best: EpochResult
    y_true: np.ndarray
    y_prob: np.ndarray
    run_time: float
    run: int = 0
    params: Dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return int(self.y_prob.shape[1])

    def __repr__(self) -> str:
        return (
            f"RunResult(dataset={self.dataset}, method={self.method}, run={self.run}, "
            f"seed={self.seed}, best_epoch={self.best.epoch}/{len(self.epochs)}, "
            f"run_time={self.run_time:.1f}s)\n{self.best}"
        )


def selection_key(
    result: EpochResult, monitor: str, split: str = "val"
) -> Tuple[float, float]:
    value = result.metric(monitor, split)
    value = -value if monitor in LOWER_IS_BETTER else value
    loss = result.metric("loss", split)
    return (
        -np.inf if np.isnan(value) else value,
        -np.inf if np.isnan(loss) else -loss,
    )


def select_best(
    results: Sequence[EpochResult], monitor: str = "roc_auc", split: str = "val"
) -> EpochResult:
    return max(results, key=lambda r: selection_key(r, monitor, split))


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


class Scoring:
    """Metrics for binary and multi-class brain network classification.

    Binary: precision, sensitivity, specificity, npv and f1 refer to class 1.
    Multi-class: precision and f1 are macro averages, roc_auc is one-vs-rest macro,
    and the binary-only metrics (pr_auc, sensitivity, specificity, npv) are NaN.
    """

    def __init__(self, num_classes: int, monitor: str = "roc_auc"):
        if monitor not in METRICS:
            raise ValueError(f"monitor must be one of {METRICS}")
        self.num_classes = num_classes
        self.monitor = monitor

    @property
    def binary(self) -> bool:
        return self.num_classes == 2

    @property
    def available(self) -> Tuple[str, ...]:
        if self.binary:
            return METRICS
        return tuple(m for m in METRICS if m not in BINARY_ONLY)

    @staticmethod
    def probabilities(logits) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def nll(y: np.ndarray, p: np.ndarray) -> float:
        return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))

    def roc_auc(self, y: np.ndarray, p: np.ndarray) -> float:
        present = np.unique(y)
        if len(present) < 2:
            return np.nan
        if self.binary:
            return float(roc_auc_score(y, p[:, 1]))
        return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in present]))

    @staticmethod
    def binary_scores(y: np.ndarray, pred: np.ndarray, p: np.ndarray) -> Dict:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        both = len(np.unique(y)) == 2
        return {
            "sensitivity": _ratio(tp, tp + fn),
            "specificity": _ratio(tn, tn + fp),
            "precision": _ratio(tp, tp + fp),
            "npv": _ratio(tn, tn + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "pr_auc": average_precision_score(y, p[:, 1]) if both else np.nan,
        }

    def score(self, y_true, logits, loss: Optional[float] = None) -> Dict[str, float]:
        y = np.asarray(y_true, dtype=np.int64).reshape(-1)
        p = self.probabilities(logits)
        pred = p.argmax(axis=1)
        s = dict.fromkeys(METRICS, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s["loss"] = self.nll(y, p) if loss is None else float(loss)
            s["accuracy"] = accuracy_score(y, pred)
            s["balanced_accuracy"] = balanced_accuracy_score(y, pred)
            s["mcc"] = matthews_corrcoef(y, pred)
            s["roc_auc"] = self.roc_auc(y, p)
            if self.binary:
                s.update(self.binary_scores(y, pred, p))
            else:
                labels = list(range(self.num_classes))
                s["precision"] = precision_score(
                    y, pred, labels=labels, average="macro", zero_division=0
                )
                s["f1"] = f1_score(
                    y, pred, labels=labels, average="macro", zero_division=0
                )
        return {k: float(v) for k, v in s.items()}

    def score_epoch(
        self,
        epoch: int,
        method: str,
        dataset: str,
        seed: int,
        duration: float,
        outputs: TTV,
    ) -> EpochResult:
        """outputs.train/val/test are (y_true, logits or log-probs, loss or None)."""
        scores = {s: self.score(*getattr(outputs, s)) for s in SPLITS}
        metrics = {
            m: TTV(
                train=scores["train"][m], val=scores["val"][m], test=scores["test"][m]
            )
            for m in METRICS
        }
        return EpochResult(
            dataset=dataset,
            method=method,
            seed=seed,
            epoch=epoch,
            duration=duration,
            **metrics,
        )

    def is_better(self, result: EpochResult, best: Optional[EpochResult]) -> bool:
        if best is None:
            return True
        return selection_key(result, self.monitor) > selection_key(best, self.monitor)

    def best(
        self,
        results: Sequence[EpochResult],
        monitor: Optional[str] = None,
        split: str = "val",
    ) -> EpochResult:
        return select_best(results, monitor or self.monitor, split)
