import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, roc_curve

from .scoring import (
    LOWER_IS_BETTER,
    METRICS,
    SPLITS,
    EpochResult,
    RunResult,
    select_best,
)


def flatten_runs(runs) -> List[RunResult]:
    if isinstance(runs, RunResult):
        return [runs]
    if isinstance(runs, dict):
        runs = list(runs.values())
    return [r for item in runs for r in flatten_runs(item)]


def flatten_dict(values: Dict, prefix: str = "") -> Dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, dict):
            out.update(flatten_dict(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = str(v) if isinstance(v, (list, tuple)) else v
    return out


def holm(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def _fmt(mean: float, std: float, decimals: int) -> str:
    if np.isnan(mean):
        return "-"
    if np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


class Affichage:
    """Tables, Excel exports and figures for evaluation results."""

    def __init__(self, outdir: str = "./exports", show: bool = True, dpi: int = 150):
        self.outdir = Path(outdir)
        self.show = show
        self.dpi = dpi

    def path(self, name: str) -> Path:
        path = Path(name)
        if not path.is_absolute() and path.parent == Path("."):
            path = self.outdir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def epochs_df(self, runs) -> pd.DataFrame:
        rows = []
        for run in flatten_runs(runs):
            for e in run.epochs:
                row = e.to_dict()
                head = {"dataset": row.pop("dataset"), "method": row.pop("method")}
                rows.append({**head, "run": run.run, **row})
        return pd.DataFrame(rows)

    def evaluation_results_to_df(
        self,
        results,
        outname: Optional[str] = None,
        export: bool = False,
    ) -> pd.DataFrame:
        if len(results) and isinstance(results[0], EpochResult):
            df = pd.DataFrame([e.to_dict() for e in results])
        else:
            df = self.epochs_df(results)
        if export and outname:
            df.to_excel(self.path(outname), index=False)
        return df

    def runs_df(self, runs, monitor: Optional[str] = None) -> pd.DataFrame:
        """One row per run: every metric of the selected (best validation) epoch."""
        rows = []
        for run in flatten_runs(runs):
            best = run.best if monitor is None else select_best(run.epochs, monitor)
            row = {
                "dataset": run.dataset,
                "method": run.method,
                "run": run.run,
                "seed": run.seed,
                "best_epoch": best.epoch,
                "epochs": len(run.epochs),
                "epoch_time": float(np.mean([e.duration for e in run.epochs])),
                "run_time": run.run_time,
            }
            row.update({f"{m}_{s}": best.metric(m, s) for m in METRICS for s in SPLITS})
            rows.append(row)
        return pd.DataFrame(rows)

    def summary_table(
        self,
        runs,
        split: str = "test",
        metrics: Sequence[str] = METRICS,
        monitor: Optional[str] = None,
    ) -> pd.DataFrame:
        """Mean and standard deviation over the runs of every dataset and method."""
        df = self.runs_df(runs, monitor)
        groups = df.groupby(["dataset", "method"], sort=False)
        out = groups.agg(
            runs=("run", "count"),
            best_epoch=("best_epoch", "mean"),
            epoch_time=("epoch_time", "mean"),
            run_time=("run_time", "mean"),
        )
        for m in metrics:
            out[f"{m}_mean"] = groups[f"{m}_{split}"].mean()
            out[f"{m}_std"] = groups[f"{m}_{split}"].std()
        return out.dropna(axis=1, how="all").reset_index()

    def summary_text(
        self,
        runs,
        split: str = "test",
        metrics: Sequence[str] = METRICS,
        decimals: int = 3,
        percent: bool = False,
        monitor: Optional[str] = None,
    ) -> pd.DataFrame:
        """Same as summary_table, formatted as 'mean ± std' for papers."""
        table = self.summary_table(runs, split, metrics, monitor)
        out = table[["dataset", "method", "runs"]].copy()
        for m in metrics:
            if f"{m}_mean" not in table:
                continue
            scale = 100 if percent and m != "loss" else 1
            mean = table[f"{m}_mean"] * scale
            std = table[f"{m}_std"] * scale if f"{m}_std" in table else mean * np.nan
            out[m] = [_fmt(a, b, decimals) for a, b in zip(mean, std)]
        return out

    def paper_table(
        self,
        runs,
        metric: str = "accuracy",
        split: str = "test",
        decimals: int = 3,
        percent: bool = False,
        monitor: Optional[str] = None,
    ) -> pd.DataFrame:
        """Methods x datasets table of one metric ('mean ± std')."""
        text = self.summary_text(runs, split, (metric,), decimals, percent, monitor)
        table = text.pivot(index="method", columns="dataset", values=metric)
        table = table.reindex(
            index=list(dict.fromkeys(text["method"])),
            columns=list(dict.fromkeys(text["dataset"])),
        )
        table.columns.name = None
        return table.reset_index()

    def significance_table(
        self,
        runs,
        metric: str = "roc_auc",
        split: str = "test",
        reference: Optional[str] = None,
        monitor: Optional[str] = None,
    ) -> pd.DataFrame:
        """Paired Wilcoxon tests against a reference method, Holm-corrected per dataset.

        The reference defaults to the best mean. Runs are paired by run index, so
        compare methods evaluated on the same seeds or folds.
        """
        df = self.runs_df(runs, monitor)
        rows = []
        for dataset, d in df.groupby("dataset", sort=False):
            table = d.pivot_table(
                index="run", columns="method", values=f"{metric}_{split}"
            )
            table = table[[m for m in dict.fromkeys(d["method"]) if m in table]]
            means = table.mean()
            ref = reference
            if ref not in table:
                ref = means.idxmin() if metric in LOWER_IS_BETTER else means.idxmax()
            block = []
            for method in table.columns:
                if method == ref:
                    continue
                pair = table[[ref, method]].dropna()
                diff = pair[method] - pair[ref]
                p = np.nan
                if len(pair) > 1 and np.any(diff != 0):
                    try:
                        p = float(wilcoxon(diff).pvalue)
                    except ValueError:
                        pass
                block.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reference": ref,
                        "pairs": len(pair),
                        f"{metric}_mean": pair[method].mean(),
                        f"reference_{metric}_mean": pair[ref].mean(),
                        "mean_diff": diff.mean(),
                        "p_value": p,
                    }
                )
            for row, adjusted in zip(block, holm([r["p_value"] for r in block])):
                row["p_holm"] = adjusted
            rows += block
        return pd.DataFrame(rows)

    def params_df(self, runs) -> pd.DataFrame:
        rows, seen = [], set()
        for run in flatten_runs(runs):
            if (run.dataset, run.method) in seen:
                continue
            seen.add((run.dataset, run.method))
            params = {k: v for k, v in flatten_dict(run.params).items() if k != "seed"}
            rows.append({"dataset": run.dataset, "method": run.method, **params})
        return pd.DataFrame(rows)

    def export_excel(
        self,
        outname: str,
        runs=None,
        searches=None,
        split: str = "test",
        monitor: Optional[str] = None,
        sheets: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Path:
        """Workbook: summaries, best epoch per run, every epoch, params, searches."""
        frames = []
        if runs is not None:
            frames += [
                ("summary", self.summary_table(runs, split, monitor=monitor)),
                ("summary_text", self.summary_text(runs, split, monitor=monitor)),
                ("runs", self.runs_df(runs, monitor)),
                ("epochs", self.epochs_df(runs)),
                ("params", self.params_df(runs)),
            ]
        frames += [(f"search_{s.dataset}_{s.method}", s.trials) for s in searches or []]
        frames += list((sheets or {}).items())
        path = self.path(outname)
        used = set()
        with pd.ExcelWriter(path) as writer:
            for name, df in frames:
                sheet = base = re.sub(r"[\[\]:*?/\\]", "-", name)[:31]
                while sheet in used:
                    sheet = f"{base[:27]}_{len(used)}"
                used.add(sheet)
                df.to_excel(writer, sheet_name=sheet, index=False)
        return path

    def grid(self, n: int, ncols: int = 3, size=(5.0, 3.8)):
        ncols = max(1, min(n, ncols))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False
        )
        axes = axes.ravel()
        for ax in axes[n:]:
            ax.set_visible(False)
        return fig, axes[:n]

    def finish(self, fig, outname: Optional[str] = None):
        fig.tight_layout()
        if outname:
            fig.savefig(self.path(outname), dpi=self.dpi, bbox_inches="tight")
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_learning_curves(
        self,
        runs,
        metric: str = "loss",
        splits: Sequence[str] = SPLITS,
        outname: Optional[str] = None,
    ):
        """Mean ± std over runs at every epoch, one panel per dataset and method."""
        df = self.epochs_df(runs)
        groups = list(df.groupby(["dataset", "method"], sort=False))
        fig, axes = self.grid(len(groups))
        for ax, ((dataset, method), d) in zip(axes, groups):
            stats = d.groupby("epoch")
            for split in splits:
                mean = stats[f"{metric}_{split}"].mean()
                std = stats[f"{metric}_{split}"].std().fillna(0)
                ax.plot(mean.index, mean.values, label=split)
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
            ax.set(title=f"{dataset} | {method}", xlabel="epoch", ylabel=metric)
            ax.legend(frameon=False)
        return self.finish(fig, outname)

    def plot_comparison(
        self,
        runs,
        metric: str = "roc_auc",
        split: str = "test",
        outname: Optional[str] = None,
        monitor: Optional[str] = None,
    ):
        """One metric over runs for every method, one panel per dataset."""
        df = self.runs_df(runs, monitor)
        col = f"{metric}_{split}"
        datasets = list(dict.fromkeys(df["dataset"]))
        width = max(4.5, 0.9 * df["method"].nunique() + 2)
        fig, axes = self.grid(len(datasets), size=(width, 4.0))
        rng = np.random.default_rng(0)
        for ax, dataset in zip(axes, datasets):
            d = df[df["dataset"] == dataset]
            methods = list(dict.fromkeys(d["method"]))
            values = [d.loc[d["method"] == m, col].dropna().values for m in methods]
            ax.boxplot(
                [v if len(v) else [np.nan] for v in values],
                showmeans=True,
                widths=0.5,
                medianprops=dict(color="C1"),
                meanprops=dict(
                    marker="D", markerfacecolor="white", markeredgecolor="C0"
                ),
            )
            for i, v in enumerate(values, start=1):
                jitter = rng.uniform(-0.12, 0.12, len(v))
                ax.scatter(i + jitter, v, s=12, alpha=0.6, color="0.35", zorder=3)
            ax.set_xticks(range(1, len(methods) + 1), methods, rotation=30, ha="right")
            ax.set(title=dataset, ylabel=f"{metric} ({split})")
        return self.finish(fig, outname)

    def plot_roc_curves(self, runs, outname: Optional[str] = None):
        """Mean ROC curve over runs for every method, binary tasks only."""
        binary = [r for r in flatten_runs(runs) if r.num_classes == 2]
        if not binary:
            raise ValueError("ROC curves are drawn for binary tasks only")
        datasets = list(dict.fromkeys(r.dataset for r in binary))
        fpr_grid = np.linspace(0, 1, 201)
        fig, axes = self.grid(len(datasets), size=(4.8, 4.4))
        for ax, dataset in zip(axes, datasets):
            by_method: Dict[str, List[RunResult]] = {}
            for r in binary:
                if r.dataset == dataset:
                    by_method.setdefault(r.method, []).append(r)
            for method, rs in by_method.items():
                tprs, aucs = [], []
                for r in rs:
                    if len(np.unique(r.y_true)) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(r.y_true, r.y_prob[:, 1])
                    tprs.append(np.interp(fpr_grid, fpr, tpr))
                    tprs[-1][0] = 0.0
                    aucs.append(r.best.roc_auc.test)
                if not tprs:
                    continue
                mean, std = np.mean(tprs, axis=0), np.std(tprs, axis=0)
                (line,) = ax.plot(
                    fpr_grid,
                    mean,
                    label=f"{method} ({np.mean(aucs):.3f} ± {np.std(aucs):.3f})",
                )
                ax.fill_between(
                    fpr_grid,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    color=line.get_color(),
                    alpha=0.12,
                )
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6")
            ax.set(
                title=dataset,
                xlabel="false positive rate",
                ylabel="true positive rate",
                xlim=(0, 1),
                ylim=(0, 1.01),
            )
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        return self.finish(fig, outname)

    def plot_confusion_matrices(
        self, runs, normalize: bool = True, outname: Optional[str] = None
    ):
        """Confusion matrix at the best-val epoch, summed over runs."""
        groups: Dict = {}
        for r in flatten_runs(runs):
            groups.setdefault((r.dataset, r.method), []).append(r)
        fig, axes = self.grid(len(groups), ncols=4, size=(3.8, 3.4))
        for ax, ((dataset, method), rs) in zip(axes, groups.items()):
            labels = list(range(rs[0].num_classes))
            cm = sum(
                confusion_matrix(r.y_true, r.y_prob.argmax(axis=1), labels=labels)
                for r in rs
            ).astype(float)
            if normalize:
                cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            sns.heatmap(
                cm,
                annot=len(labels) <= 10,
                fmt=".2f" if normalize else ".0f",
                cmap="Blues",
                vmin=0,
                vmax=1 if normalize else None,
                cbar=False,
                square=True,
                ax=ax,
            )
            ax.set(title=f"{dataset} | {method}", xlabel="predicted", ylabel="true")
        return self.finish(fig, outname)

    def plot_runtime(self, runs, log: bool = False, outname: Optional[str] = None):
        """Mean training time per epoch for every method, one panel per dataset."""
        df = self.runs_df(runs)
        datasets = list(dict.fromkeys(df["dataset"]))
        width = max(4.5, 0.9 * df["method"].nunique() + 2)
        fig, axes = self.grid(len(datasets), size=(width, 3.8))
        for ax, dataset in zip(axes, datasets):
            d = df[df["dataset"] == dataset]
            times = d.groupby("method", sort=False)["epoch_time"]
            mean, std = times.mean(), times.std().fillna(0)
            x = range(len(mean))
            ax.bar(x, mean.values, yerr=std.values, capsize=3, color="C0")
            ax.set_xticks(x, mean.index, rotation=30, ha="right")
            ax.set(title=dataset, ylabel="seconds per epoch")
            if log:
                ax.set_yscale("log")
        return self.finish(fig, outname)

    def plot_search(self, search, outname: Optional[str] = None):
        """Validation score of the searched combinations, per hyperparameter value."""
        trials = search.trials
        score = f"val_{search.monitor}"
        params = [
            c
            for c in trials.columns
            if c not in (score, f"{score}_std", "best_epoch", "run_time")
        ]
        params = [c for c in params if trials[c].nunique() > 1] or params
        fig, axes = self.grid(len(params), size=(4.2, 3.4))
        for ax, p in zip(axes, params):
            values = list(dict.fromkeys(trials[p]))
            try:
                values = sorted(values)
            except TypeError:
                pass
            data = [trials.loc[trials[p] == v, score].dropna().values for v in values]
            ax.boxplot([d if len(d) else [np.nan] for d in data], widths=0.5)
            ticks = range(1, len(values) + 1)
            labels = [str(v) for v in values]
            ax.set_xticks(ticks, labels, rotation=30, ha="right")
            ax.set(title=p, ylabel=score)
        fig.suptitle(f"{search.dataset} | {search.method}: {len(trials)} trials")
        return self.finish(fig, outname)

    def plot_adjacency(
        self,
        A: np.ndarray,
        title: str = "adjacency matrix",
        outname: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            np.asarray(A),
            cmap="Greys",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(title)
        return self.finish(fig, outname)

    def plot_connectome(
        self,
        fc: np.ndarray,
        A: Optional[np.ndarray] = None,
        title: str = "",
        outname: Optional[str] = None,
    ):
        """Connectivity of one subject, with its thresholded adjacency if given."""
        fig, axes = self.grid(1 if A is None else 2, ncols=2, size=(5.4, 4.6))
        fc = np.asarray(fc)
        vmax = float(np.nanmax(np.abs(fc))) or 1.0
        sns.heatmap(
            fc,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=axes[0],
        )
        axes[0].set_title(f"{title} connectivity".strip())
        if A is not None:
            sns.heatmap(
                np.asarray(A) != 0,
                cmap="Greys",
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
                ax=axes[1],
            )
            density = float(np.mean(np.asarray(A) != 0))
            axes[1].set_title(f"{title} adjacency ({density:.1%} edges)".strip())
        return self.finish(fig, outname)

    def report(
        self,
        runs,
        name: str,
        searches=None,
        metric: str = "roc_auc",
        monitor: Optional[str] = None,
    ) -> Path:
        """Excel workbook plus the standard figures for a set of runs."""
        path = self.export_excel(f"{name}.xlsx", runs, searches, monitor=monitor)
        show, self.show = self.show, False
        try:
            self.plot_comparison(
                runs, metric, outname=f"{name}_{metric}.png", monitor=monitor
            )
            self.plot_learning_curves(runs, "loss", outname=f"{name}_loss.png")
            self.plot_learning_curves(
                runs, metric, outname=f"{name}_{metric}_curves.png"
            )
            self.plot_confusion_matrices(runs, outname=f"{name}_confusion.png")
            if any(r.num_classes == 2 for r in flatten_runs(runs)):
                self.plot_roc_curves(runs, outname=f"{name}_roc.png")
            self.plot_runtime(runs, outname=f"{name}_runtime.png")
            for s in searches or []:
                self.plot_search(s, outname=f"{name}_search_{s.method}.png")
        finally:
            self.show = show
        return path
