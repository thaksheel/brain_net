import copy, itertools, json, random, re 
from dataclasses import dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from .base_trainer import BaseTrainer, apply_params, params_from_dict, params_to_dict
from .config import TTV, Params
from .graph_trainer import GraphTrainer
from .node_trainer import NodeTrainer
from .scoring import LOWER_IS_BETTER, RunResult

STRUCTURE_KEYS = (
    "Mlst",
    "num_layers",
    "self_loop",
    "tmphn_config.hyperedges",
    "tmphn_config.strategy",
    "tmphn_config.knn_k",
    "tmphn_config.scores",
    "tmphn_config.temperature",
)


def _cell(value: Any) -> Any:
    return str(value) if isinstance(value, (list, tuple, dict)) else value


@dataclass
class SearchResult:
    dataset: str
    method: str
    monitor: str
    best_params: Dict[str, Any]
    best_score: float
    trials: pd.DataFrame
    path: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"SearchResult(dataset={self.dataset}, method={self.method}, "
            f"trials={len(self.trials)}, val_{self.monitor}={self.best_score:.4f}, "
            f"best_params={self.best_params}, path={self.path})"
        )


class Evaluation:
    """Hyperparameter search, repeated evaluation and saved parameter sets."""

    def __init__(
        self,
        params: Params,
        task: Literal["graph", "node"] = "graph",
        monitor: str = "roc_auc",
        params_dir: str = "./params",
        display: bool = False,
        progress_bar: bool = True,
        verbose: bool = True,
        patience: Optional[int] = None,
        class_weight: Optional[str] = None,
    ):
        if task not in ("graph", "node"):
            raise ValueError("task must be 'graph' or 'node'")
        self.params = params
        self.task = task
        self.monitor = monitor
        self.params_dir = Path(params_dir)
        self.verbose = verbose
        self.trainer_kwargs = dict(
            display=display,
            progress_bar=progress_bar,
            monitor=monitor,
            patience=patience,
            class_weight=class_weight,
        )
        self.runs: List[RunResult] = []
        self.searches: List[SearchResult] = []

    @property
    def dataset_name(self) -> str:
        return str(self.params.dataset)

    def trainer(self, params: Params, **overrides) -> BaseTrainer:
        Trainer = GraphTrainer if self.task == "graph" else NodeTrainer
        return Trainer(copy.deepcopy(params), **{**self.trainer_kwargs, **overrides})

    def for_method(
        self, method: Optional[str] = None, params: Optional[Params] = None
    ) -> Params:
        params = copy.deepcopy(params or self.params)
        params.method = method or params.method
        return params

    def better(self, score: float, best: Optional[float]) -> bool:
        if best is None or np.isnan(best):
            return True
        if np.isnan(score):
            return False
        return score < best if self.monitor in LOWER_IS_BETTER else score > best

    @staticmethod
    def free_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def grid_dict(grid) -> Dict[str, List]:
        if is_dataclass(grid):
            grid = vars(grid)
        return {
            k: list(v) if isinstance(v, (list, tuple, range)) else [v]
            for k, v in grid.items()
            if v is not None
        }

    def params_path(self, method: str, dataset: Optional[str] = None) -> Path:
        name = f"{dataset or self.dataset_name}_{method}"
        return self.params_dir / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', name)}.json"

    def save_params(
        self,
        params: Params,
        search: Optional[SearchResult] = None,
        path: Optional[str] = None,
    ) -> Path:
        """Store the parameters of a full evaluation as JSON to rerun it later."""
        if path is None:
            path = self.params_path(params.method, str(params.dataset))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(params.dataset),
            "method": params.method,
            "task": self.task,
            "monitor": self.monitor,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "params": params_to_dict(params),
        }
        if search is not None:
            payload["search"] = {
                f"val_{search.monitor}": search.best_score,
                "best_params": params_to_dict(search.best_params),
                "trials": len(search.trials),
            }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def load_params(
        self,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Params:
        if path is None:
            path = self.params_path(method or self.params.method, dataset)
        path = Path(path)
        params = params_from_dict(json.loads(path.read_text())["params"])
        params.device = self.params.device
        return params

    def has_saved(self, method: str, dataset: Optional[str] = None) -> bool:
        return self.params_path(method, dataset).exists()

    def grid_search(
        self,
        data,
        method: Optional[str] = None,
        grid=None,
        maxiter: Optional[int] = 50,
        epochs: Optional[int] = None,
        seeds: Optional[Iterable[int]] = None,
        save: bool = True,
    ) -> SearchResult:
        """Score `maxiter` distinct grid combinations on the validation split only.

        grid: GridSearchParams or dict of candidate lists; nested keys use dots,
        e.g. {"tmphn_config.hyperedges": ["neighborhood", "knn"]}.
        seeds: average the validation score over several splits (noisy small datasets).
        """
        base = self.for_method(method)
        if epochs is not None:
            base.epochs = epochs
        grid = self.grid_dict(grid)
        keys = list(grid)
        combos = list(itertools.product(*grid.values()))
        if maxiter is not None and maxiter < len(combos):
            combos = random.Random(base.seed).sample(combos, maxiter)
        structural = [i for i, k in enumerate(keys) if k in STRUCTURE_KEYS]
        combos.sort(key=lambda c: [repr(c[i]) for i in structural])
        seeds = list(seeds) if seeds is not None else [base.seed]
        trainer = self.trainer(base, progress_bar=False, display=False)
        score_col = f"val_{self.monitor}"
        rows, best, best_score = [], None, None
        for combo in tqdm(
            combos,
            desc=f"search|{self.dataset_name}|{base.method}",
            disable=not self.trainer_kwargs["progress_bar"],
        ):
            overrides = dict(zip(keys, combo))
            runs = []
            for seed in seeds:
                trainer.params = apply_params(base, {**overrides, "seed": seed})
                runs.append(trainer.train(data, base.method))
                self.free_memory()
            scores = np.array([r.best.metric(self.monitor, "val") for r in runs])
            finite = np.isfinite(scores).any()
            score = float(np.nanmean(scores)) if finite else np.nan
            spread = float(np.nanstd(scores)) if finite and len(seeds) > 1 else np.nan
            rows.append(
                {
                    **{k: _cell(v) for k, v in overrides.items()},
                    score_col: score,
                    f"{score_col}_std": spread,
                    "best_epoch": float(np.mean([r.best.epoch for r in runs])),
                    "run_time": float(sum(r.run_time for r in runs)),
                }
            )
            if self.better(score, best_score):
                best, best_score = overrides, score
        trials = (
            pd.DataFrame(rows)
            .sort_values(
                score_col,
                ascending=self.monitor in LOWER_IS_BETTER,
                na_position="last",
            )
            .reset_index(drop=True)
        )
        result = SearchResult(
            dataset=self.dataset_name,
            method=base.method,
            monitor=self.monitor,
            best_params=best,
            best_score=best_score,
            trials=trials,
        )
        if save:
            params = apply_params(self.for_method(base.method), best)
            result.path = str(self.save_params(params, search=result))
        self.searches.append(result)
        if self.verbose:
            print(result)
        return result

    def exhaustive_search(
        self,
        data,
        method: Optional[str] = None,
        grid=None,
        epochs: Optional[int] = None,
        seeds: Optional[Iterable[int]] = None,
        save: bool = True,
    ) -> SearchResult:
        """Every combination of the grid."""
        return self.grid_search(data, method, grid, None, epochs, seeds, save)

    def report(self, run: RunResult, i: int, n: int):
        if not self.verbose:
            return
        b, m = run.best, self.monitor
        print(
            f"[{run.dataset}|{run.method}] run {i + 1}/{n} seed={run.seed} "
            f"best_epoch={b.epoch}/{len(run.epochs)} val_{m}={b.metric(m, 'val'):.4f} "
            f"test_{m}={b.metric(m, 'test'):.4f} test_acc={b.accuracy.test:.4f} "
            f"time={run.run_time:.1f}s"
        )

    def run_stratified_n_iterations(
        self,
        data,
        method: Optional[str] = None,
        seeds: Optional[Iterable[int]] = None,
        n: Optional[int] = None,
        params: Optional[Params] = None,
    ) -> List[RunResult]:
        """Repeat the stratified train/val/test split over seeds (one run per seed).

        Each run reports its test scores at the epoch with the best validation score.
        """
        params = self.for_method(method, params)
        if seeds is None:
            seeds = range(params.seed, params.seed + (n or params.num_exps))
        seeds = list(seeds)
        trainer = self.trainer(params)
        runs = []
        for i, seed in enumerate(seeds):
            trainer.params.seed = seed
            run = trainer.train(data, params.method)
            run.run = i
            runs.append(run)
            self.report(run, i, len(seeds))
            self.free_memory()
        self.runs.extend(runs)
        return runs

    def cross_validate(
        self,
        data,
        method: Optional[str] = None,
        folds: int = 5,
        repeats: int = 1,
        params: Optional[Params] = None,
    ) -> List[RunResult]:
        """Stratified k-fold: each subject is tested once per repeat.

        Validation subjects are drawn from the training folds.
        """
        params = self.for_method(method, params)
        trainer = self.trainer(params)
        labels = trainer.labels(data)
        val_size = params.valid_ratio / (1 - 1 / folds)
        runs = []
        for r in range(repeats):
            kfold = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=params.seed + r
            )
            folds_idx = kfold.split(np.zeros(len(labels)), labels)
            for k, (rest, test) in enumerate(folds_idx):
                train, val = train_test_split(
                    rest,
                    test_size=val_size,
                    stratify=labels[rest],
                    random_state=params.seed + r,
                )
                trainer.params.seed = params.seed + r * folds + k
                run = trainer.train(
                    data, params.method, split=TTV(train=train, val=val, test=test)
                )
                run.run = r * folds + k
                runs.append(run)
                self.report(run, run.run, folds * repeats)
                self.free_memory()
        self.runs.extend(runs)
        return runs

    def tune_and_evaluate(
        self,
        data,
        method: str,
        grid,
        seeds: Optional[Iterable[int]] = None,
        n: Optional[int] = None,
        maxiter: Optional[int] = 50,
        search_epochs: Optional[int] = None,
        search_seeds: Optional[Iterable[int]] = None,
    ) -> List[RunResult]:
        """Grid search, save the best parameters, then repeat over seeds."""
        search = self.grid_search(
            data, method, grid, maxiter, search_epochs, search_seeds, save=True
        )
        params = apply_params(self.for_method(method), search.best_params)
        return self.run_stratified_n_iterations(data, method, seeds, n, params)

    def evaluate_saved(
        self,
        data,
        method: str,
        seeds: Optional[Iterable[int]] = None,
        n: Optional[int] = None,
        path: Optional[str] = None,
    ) -> List[RunResult]:
        """Rerun a full evaluation from parameters saved by a previous search."""
        params = self.load_params(method, path=path)
        return self.run_stratified_n_iterations(data, method, seeds, n, params)

    def compare(
        self,
        data,
        methods: Sequence[str],
        seeds: Optional[Iterable[int]] = None,
        n: Optional[int] = None,
        use_saved: bool = True,
    ) -> List[RunResult]:
        """Evaluate methods on the same seeds (paired runs), saved params if any."""
        seeds = None if seeds is None else list(seeds)
        runs = []
        for method in methods:
            saved = use_saved and self.has_saved(method)
            params = self.load_params(method) if saved else self.for_method(method)
            if self.verbose:
                source = self.params_path(method) if saved else "base params"
                print(f"[{self.dataset_name}|{method}] using {source}")
            runs += self.run_stratified_n_iterations(data, method, seeds, n, params)
        return runs
