import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
import matplotlib.pyplot as plt
import itertools
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    f1_score,
)
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
import os
import random
import time
from typing import List, Tuple, Dict

from . import (
    NeuroGraphDataset,
    fix_seed,
    EvalResults,
    TTV,
    NeuroGraphParams,
    ResidualGNNs,
)


class NGEstimator:
    def __init__(self, params: NeuroGraphParams, display: bool = True):
        self.params = params
        self.display = display

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.params, k, v)
        return self

    def initialize(
        self,
        root_folder: str,
        dataset_name: str,
    ) -> NeuroGraphDataset:
        dataset = NeuroGraphDataset(root=root_folder, name=dataset_name)
        self.params.num_features = dataset.num_features
        self.params.num_classes = dataset.num_classes
        return dataset

    def ttv_splits(self, dataset: NeuroGraphDataset) -> TTV:
        labels = [d.y.item() for d in dataset]
        train_tmp, test_indices = train_test_split(
            list(range(len(labels))),
            test_size=0.2,
            stratify=labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        tmp = dataset[train_tmp]
        train_labels = [d.y.item() for d in tmp]
        train_indices, val_indices = train_test_split(
            list(range(len(train_labels))),
            test_size=0.125,
            stratify=train_labels,
            random_state=self.params.seed,
            shuffle=True,
        )
        train_dataset = tmp[train_indices]
        val_dataset = tmp[val_indices]
        test_dataset = dataset[test_indices]
        return TTV(train=train_dataset, test=test_dataset, val=val_dataset)

    def get_dataloader(self, ttv_dataset: TTV) -> TTV:
        train_loader = DataLoader(
            ttv_dataset.train, self.params.batch_size, shuffle=False
        )
        val_loader = DataLoader(ttv_dataset.val, self.params.batch_size, shuffle=False)
        test_loader = DataLoader(
            ttv_dataset.test, self.params.batch_size, shuffle=False
        )
        return TTV(train=train_loader, val=val_loader, test=test_loader)

    def get_base_gnn(self):
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
        return base_models[self.params.model]

    def get_eval_results(
        self,
        epoch,
        method,
        duration,
        train_loss,
        test_loss,
        val_loss,
        train_true,
        train_pred,
        test_true,
        test_pred,
        val_true,
        val_pred,
    ):
        return EvalResults(
            epoch=epoch,
            method=method,
            duration=duration,
            accuracy=TTV(
                train=accuracy_score(train_true, train_pred),
                test=accuracy_score(test_true, test_pred),
                val=accuracy_score(val_true, val_pred),
            ),
            f1=TTV(
                train=f1_score(train_true, train_pred, average="macro"),
                test=f1_score(test_true, test_pred, average="macro"),
                val=f1_score(val_true, val_pred, average="macro"),
            ),
            loss=TTV(
                train=train_loss,
                test=test_loss,
                val=val_loss,
            ),
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
            r2=TTV(
                train=r2_score(train_true, train_pred),
                test=r2_score(test_true, test_pred),
                val=r2_score(val_true, val_pred),
            ),
            M=None,
            model=None,
        )

    def _train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: Adam,
    ):
        model.train()
        losses = []
        trues = []
        preds = []
        for data in train_loader:
            data = data.to(self.params.device)
            trues.extend(data.y)
            out = model(data)
            pred = out.argmax(dim=1)
            preds.extend(pred)
            loss = criterion(out, data.y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        trues = [t.item() for t in trues]
        preds = [t.item() for t in preds]
        return trues, preds, np.mean(losses)

    @torch.no_grad()
    def _test(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
    ):
        model.eval()
        losses = []
        trues = []
        preds = []
        for data in loader:
            data = data.to(self.params.device)
            trues.extend(data.y)
            out = model(data)
            pred = out.argmax(dim=1)
            preds.extend(pred)
            loss = criterion(out, data.y)
            losses.append(loss.item())
        trues = [t.item() for t in trues]
        preds = [t.item() for t in preds]
        return trues, preds, np.mean(losses)

    def fit_train(self) -> Tuple[torch.nn.Module, List[EvalResults]]:
        fix_seed(self.params.seed)
        dataset = self.initialize(
            root_folder=self.params.root_folder, dataset_name=self.params.dataset_name
        )
        # TODO: add an l1loss for regression --> add sl_type to class or params to auto select
        criterion = torch.nn.CrossEntropyLoss()
        ttv_dataset = self.ttv_splits(dataset=dataset)
        ttv_loader = self.get_dataloader(ttv_dataset=ttv_dataset)
        base_gnn = self.get_base_gnn()
        model = ResidualGNNs(
            args=self.params,
            train_dataset=ttv_dataset.train,
            GNN=base_gnn,
        ).to(self.params.device)
        optimizer = Adam(
            model.parameters(), lr=self.params.lr, weight_decay=self.params.wd
        )
        best_val_acc, best_test_acc = 0, 0
        evaluation_results = []
        for epoch in range(self.params.epochs):
            start = time.time()
            tr_t, tr_p, tr_l = self._train(
                train_loader=ttv_loader.train,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
            )
            dr = time.time() - start
            te_t, te_p, te_l = self._test(
                loader=ttv_loader.val,
                model=model,
                criterion=criterion,
            )
            val_t, val_p, val_l = self._test(
                loader=ttv_loader.test,
                model=model,
                criterion=criterion,
            )
            eval_result = self.get_eval_results(
                epoch=epoch,
                method=self.params.model,
                duration=dr,
                train_loss=tr_l,
                test_loss=te_l,
                val_loss=val_l,
                train_true=tr_t,
                train_pred=tr_p,
                test_pred=te_p,
                test_true=te_t,
                val_true=val_t,
                val_pred=val_p,
            )
            if self.display:
                print(
                    f"epoch={epoch} "
                    f"train_loss={eval_result.loss.train:.4f} "
                    f"test_acc={100 * eval_result.accuracy.test:.2f}% "
                    f"test_f1={100 * eval_result.f1.test:.2f}% "
                    f"duration={eval_result.duration:.1f}s"
                )
            if eval_result.accuracy.val > best_val_acc:
                best_val_acc = eval_result.accuracy.val
            if eval_result.accuracy.test > best_test_acc:
                best_test_acc = eval_result.accuracy.test
            evaluation_results.append(eval_result)
        if self.display:
            print(
                f"best_val_acc={100 * best_val_acc:.2f}%, "
                f"best_test_acc={100*best_test_acc:.2f}%"
            )
        return model, evaluation_results

    def predict(
        self,
    ):
        """Using a given model and true labels, create predictions"""
        pass

    def evaluation_results_to_df(
        self,
        evaluation_results: List[EvalResults],
        outname: str,
        export: bool = False,
        exclude_fields: List[str] = ["model"],
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

    def quick_plot(self, epochs, data, **kwargs):
        plt.plot(epochs, data, marker="o", color="steelblue")
        plt.xlabel("epochs")
        plt.ylabel("data")
        if "model" in kwargs:
            plt.title(kwargs["model"])
        plt.show()

    def get_best_eval_results(self, eval_results: List[EvalResults]) -> EvalResults:
        test_score = np.array([er.accuracy.test for er in eval_results])
        best = np.argmax(test_score)
        return eval_results[best]

    def exhaustive_search(self, param_grid: Dict):
        keys, values = zip(*param_grid.items())
        best_score, best_params = 0, None
        result_arr = []
        param_arr = []
        combos = list(itertools.product(*values))
        for k, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            self.set_params(**params)
            train_results = self.fit_train()
            best_result = self.get_best_eval_results(train_results)
            result_arr.append(best_result)
            param_arr.append(params)
            if best_result.accuracy.test > best_score:
                best_score, best_params = best_result.accuracy.test, params
            if self.display:
                print(
                    f"---> params eval, best_results={best_score} "
                    f"best_params={best_params} "
                    f"progress={k+1}/{len(combos)}"
                )
        return np.array(result_arr), np.array(param_arr)

    def grid_search(
        self,
        param_grid: Dict,
        maxiter: int = 50,
    ) -> Tuple[Dict, float]:
        best_score, best_params = 0, None
        for _ in range(maxiter):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            self.set_params(**params)
            train_results = self.fit_train()
            best_result = self.get_best_eval_results(train_results)
            if best_result.accuracy.test > best_score:
                best_score, best_params = best_result.accuracy.test, params
        return best_params, best_score
