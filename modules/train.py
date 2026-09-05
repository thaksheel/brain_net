import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import time
from matplotlib import pyplot as plt
import random
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
)
from typing import Dict, List, Tuple, Optional, Literal
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .models.TMPHN_graph_cls import TMPHN_Graph_Cls
from .prepare import initialize, read_data, add_self_loop
from .utils.Neighbors import NeighborFinder
from .utils.datasets import NeuroGraphDataset
from .config import (
    Params,
    EvalResults,
    TTV,
    from_pygeo_to_matrices,
    HypergraphBatchLoader,
)


class THGTrainer:
    def __init__(
        self,
        params: Params,
        init_stopper: int = None,
        init_limit_rois: int = None,
        limit_X_features: int = None,
        display: bool = True,
    ):
        self.params = params
        self.display = display
        self.device = torch.device(self.params.device)
        self.init_stopper = init_stopper
        self.init_limit_rois = init_limit_rois
        self.limit_X_features = limit_X_features

        # Null
        self.data: Dict = None
        self.trained_model: torch.nn.Module = None

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.params, k, v)
        return self

    def train_results_to_df(
        self,
        train_results: list[EvalResults],
        outname: str,
        export: bool = False,
        exclude_fileds: List[str] = ["model"],
    ) -> pd.DataFrame:
        data = []
        for tr in train_results:
            d = tr.__dict__
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
                    exclude_fileds.append(k)
                result[k] = v
            data.append({k: v for k, v in result.items() if k not in exclude_fileds})
        df = pd.DataFrame(data)
        if export:
            df.to_excel(outname)
        return df

    def initialize_node(
        self,
        X: np.ndarray,
        H: np.ndarray,
        Y: np.ndarray,
        datafile: str = None,
    ):
        if datafile is not None:
            H, X, Y = read_data(datafile)
        self.params.input_dim = X.shape[1]
        self.params.num_classes = len(np.unique(Y))
        start = time.time()
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
        if self.display:
            duration = (time.time() - start) / 60
            print(f"====> initialization done, runtime={duration:.2f}min")
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
            if self.init_limit_rois:
                X = X[: self.init_limit_rois, : self.limit_X_features]
                H = H[: self.init_limit_rois, : self.init_limit_rois]
            neigh_dict = NeighborFinder(H, self.params.M).neig_for_targets(
                list(range(X.shape[0]))
            )
            H = torch.from_numpy(H).long().to(self.device)
            dataset.append(Data(x=X, adj=H, neig_dict=neigh_dict, y=Y))
            if self.init_stopper:
                if len(dataset) > self.init_stopper:
                    break
        return dataset

    def load_dataset(
        self,
        root_folder: str,
        dataset_name: str,
    ) -> NeuroGraphDataset:
        dataset = NeuroGraphDataset(root=root_folder, name=dataset_name)
        self.params.num_features = dataset.num_features
        self.params.num_classes = dataset.num_classes
        return dataset

    def ttv_splits(self, dataset: List) -> TTV:
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
        train_dataset = [tmp[i] for i in train_indices]
        val_dataset = [tmp[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        return TTV(train=train_dataset, test=test_dataset, val=val_dataset)

    def get_dataloader(self, ttv_dataset: TTV) -> TTV:
        train_loader = HypergraphBatchLoader(
            ttv_dataset.train,
            self.params.batch_size,
            shuffle=False,
        )
        val_loader = HypergraphBatchLoader(
            ttv_dataset.val,
            self.params.batch_size,
            shuffle=False,
        )
        test_loader = HypergraphBatchLoader(
            ttv_dataset.test,
            self.params.batch_size,
            shuffle=False,
        )
        return TTV(train=train_loader, val=val_loader, test=test_loader)

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
                train=f1_score(train_true, train_pred, average=None),
                test=f1_score(test_true, test_pred, average=None),
                val=f1_score(val_true, val_pred, average=None),
            ),
            f1_macro=TTV(
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
            M=None,
        )

    def evaluate_graph_cls(
        self,
        root_folder: Optional[str],
        dataset_name: Optional[str],
        graph_data: Dict = None,
    ) -> List[EvalResults]:
        """Generate dataloader, batches, optimizer, and criterion to feed into `train()` and `test()` to return `List[EvalResults]`"""
        s = time.time()
        if graph_data is None:
            ng_dataset = self.load_dataset(root_folder, dataset_name)
            Hs, Xs, Ys = from_pygeo_to_matrices(ng_dataset)
        else:
            Hs, Xs, Ys = graph_data["H"], graph_data["X"], graph_data["Y"]
        dataset = self.initialize_graph(Hs, Xs, Ys)
        print(
            f"initialization for graph_cls complete "
            f"runtime={(time.time() - s)/60:.2f}mins"
        )
        ttv_dataset = self.ttv_splits(dataset)
        ttv_loader = self.get_dataloader(ttv_dataset)
        model = TMPHN_Graph_Cls(self.params).to(self.device)
        eval_results = self.train_graph_cls(model, ttv_loader)
        return eval_results

    def optimize_graph_cls(
        self,
        loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.AdamW,
        criterion: torch.nn.CrossEntropyLoss,
    ):
        """Performs gradient descent per epoch/batch and returns [predictates, loss]"""
        out: torch.Tensor
        loss: torch.Tensor
        losses = []
        trues = []
        preds = []
        outs = []
        for batch in loader:
            out = model(batch, self.params)
            y = torch.tensor([d.y for d in batch], device=self.device)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            preds.extend(out.argmax(dim=1))
            trues.extend(y)
            losses.append(loss.item())
            outs.extend(out)
            optimizer.zero_grad()
        trues = [t.item() for t in trues]
        preds = [t.item() for t in preds]
        return trues, preds, np.mean(losses), out

    def train_graph_cls(
        self,
        model: torch.nn.Module,
        ttv_loader: TTV,
    ):
        """Train GNN from loader/batches to return predictates and embeddings"""
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.params.lr, weight_decay=self.params.wd
        )
        criterion = torch.nn.CrossEntropyLoss()
        results: List[EvalResults] = []
        for epoch in range(self.params.epochs):
            start = time.time()
            trt, trp, tr_loss, out_agg = self.optimize_graph_cls(
                ttv_loader.train, model, optimizer, criterion
            )
            duration = time.time() - start
            tet, tep, te_loss = self.test_graph_cls(model, ttv_loader.test, criterion)
            valt, valp, val_loss = self.test_graph_cls(model, ttv_loader.val, criterion)
            results.append(
                self.get_eval_results(
                    epoch,
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
                )
            )
            if self.display:
                print(
                    f"epoch={epoch+1} "
                    f"train_loss={tr_loss:.3f} "
                    f"test_accuracy={results[-1].accuracy.test*100:.2f}% "
                    f"test_f1={results[-1].f1.test} "
                    f"test_f1_macro={results[-1].f1_macro.test*100:.2f}% "
                    f"out_agg_mean={out_agg.mean():.5f} "
                    f"out_agg_std={out_agg.std():.5f} "
                    f"duration={duration:.2f}s "
                )
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        print(f"{name} grad norm: {p.grad.norm().item():.4f}")

        self.trained_model = model
        return results

    def test_graph_cls(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
    ):
        """Test GNN from checkpoint to return predictates and embeddings"""
        model.eval()
        losses = []
        trues = []
        preds = []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch, self.params)
                y = torch.tensor([d.y for d in batch], device=self.device)
                pred = out.argmax(dim=1)
                preds.extend(pred)
                trues.extend(y)
                losses.append(criterion(out, y).item())
        trues = [t.item() for t in trues]
        preds = [t.item() for t in preds]
        return trues, preds, np.mean(losses)

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
        Y = data["Y"]
        train_results: list[EvalResults] = []
        for epoch in range(self.params.epochs):
            start_epoch = time.time()
            model.train()
            optimizer.zero_grad()
            if self.params.method == "T-Spectral" or self.params.method == "T-Spatial":
                output_train = model(A, X)[idx.train]
            elif self.params.method == "T-MPHN":
                output_train = model(idx.train)
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

    def node_cls_evaluation(
        self, model: torch.nn.Module, data: Dict, idx: TTV, **kwargs
    ):
        Y = data["Y"]
        model.eval()
        if self.params.method == "T-Spectral" or self.params.method == "T-Spatial":
            A, X = data["hypergraph"], data["X"]
            output = model(A, X)  # transductive
            output_train, output_val, output_test = (
                output[idx.train],
                output[idx.val],
                output[idx.test],
            )
        elif self.params.method == "T-MPHN":
            output_train, output_val, output_test = (
                model(idx.train),
                model(idx.val),
                model(idx.test),
            )  # inductive

        prediction_train = output_train.max(1)[1].type_as(Y).cpu().numpy()
        prediction_test = output_test.max(1)[1].type_as(Y).cpu().numpy()
        prediction_val = output_val.max(1)[1].type_as(Y).cpu().numpy()
        tr_loss = F.nll_loss(output_train, Y[idx.train])
        te_loss = F.nll_loss(output_test, Y[idx.test])
        val_loss = F.nll_loss(output_val, Y[idx.val])
        tr = self.get_eval_results(
            kwargs["epoch"],
            kwargs["method"],
            kwargs["duration"],
            tr_loss,
            te_loss,
            val_loss,
            Y[idx.train].cpu().numpy(),
            prediction_train,
            Y[idx.test].cpu().numpy(),
            prediction_test,
            Y[idx.val].cpu().numpy(),
            prediction_val,
        )
        if self.display:
            print(
                f"epoch={tr.epoch} "
                f"loss={tr.loss.train:.4f} "
                f"test_acc={tr.accuracy.test*100:.2f}% "
                f"f1_test={tr.f1_macro.test*100:.2f}% "
                f"duration={tr.duration:.2f}s "
                f"method={tr.method}"
            )
        return tr

    def get_best_model(self, train_results: List[EvalResults]) -> EvalResults:
        test_score = np.array([tr.accuracy.test for tr in train_results])
        best = np.argmax(test_score)
        return train_results[best]

    def predict(self, train_results: List[EvalResults]):
        tr = self.get_best_model(train_results)
        model = tr.model
        # TODO: complete this to fit any data beyond the self.data idx
        if self.params.method == "T-Spectral" or self.params.method == "T-Spatial":
            A, X = self.data["hypergraph"], self.data["X"]
            output = model(A, X)  # transductive
        elif self.params.method == "T-MPHN":
            output = model(None)  # inductive

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
            best_result = self.get_best_model(train_results)
            result_arr.append(best_result)
            param_arr.append(params)
            if best_result.accuracy.test > best_score:
                best_score, best_params = best_result.accuracy.test, params
            if self.display:
                print(
                    f"---> params eval, best_results={best_score} best_params={best_params} progress={k+1}/{len(combos)}"
                )
        return np.array(result_arr), np.array(param_arr)

    def grid_search(self, param_grid: Dict, maxiter: int = 50):
        best_score, best_params = 0, None
        for _ in range(maxiter):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            self.set_params(**params)
            train_results = self.fit_train()
            best_result = self.get_best_model(train_results)
            if best_result.accuracy.test > best_score:
                best_score, best_params = best_result.accuracy.test, params
        return best_params, best_score

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

    def get_best_eval_results(self, eval_results: List[EvalResults]) -> EvalResults:
        test_score = np.array([er.accuracy.test for er in eval_results])
        best = np.argmax(test_score)
        return eval_results[best]
