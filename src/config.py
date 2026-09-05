import argparse
from dataclasses import dataclass, field
import numpy as np
from typing import List, Literal, Any, Dict
import torch
from torch_geometric.data import Data, InMemoryDataset
import itertools
import random


def parse():
    """
    adds and parses arguments / hyperparameters
    """
    p = argparse.ArgumentParser(
        description="TensorHyperGNNs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_type",
        type=str,
        default="new",
        help="data type (coauthorship/cotation/3dObject/new)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="House",
        help="dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation, ModelNet40/NTU for 3dObject, House/Walmart for new)",
    )
    p.add_argument(
        "--hyperG_norm",
        type=bool,
        default=False,
        help="whether normalize hypergraph adjacency tensor",
    )
    p.add_argument(
        "--model",
        type=str,
        default="T-MPHN",
        help="T-HyperGNN models(T-Spectral, T-Spatial, T-MPHN)",
    )
    p.add_argument(
        "--self_loop",
        type=bool,
        default=False,
        help="whether add self loop to hypergraph",
    )
    p.add_argument(
        "--num_layers", type=int, default=1, help="number of HyperGNN layers"
    )
    p.add_argument(
        "--hid_dim",
        type=int,
        default=256,
        help="the dimension of embeddings at the hidden layer",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="dropout probability after each HyperGNN layer",
    )
    p.add_argument(
        "--layernorm", type=bool, default=True, help="whether use layer normalization"
    )
    p.add_argument(
        "--batchnorm", type=bool, default=True, help="whether use batch normalization"
    )
    p.add_argument("--lr", type=float, default=0.001, help="learning rate")
    p.add_argument(
        "--wd", type=float, default=0.0005, help="weight decay of learning rate"
    )
    p.add_argument(
        "--train_ratio", type=float, default=0.5, help="ratio of training data"
    )
    p.add_argument(
        "--valid_ratio", type=float, default=0.25, help="ratio of validation data"
    )
    p.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    p.add_argument(
        "--num_exps", type=int, default=2, help="number of repeated experiments"
    )
    p.add_argument("--cuda", type=int, default=0, help="cuda id to use")
    p.add_argument("--seed", type=int, default=42, help="seed for randomness")
    p.add_argument(
        "--early_stopping",
        type=bool,
        default=True,
        help="early stopping after convergence",
    )
    p.add_argument(
        "--combine",
        type=str,
        default="concat",
        help="the combine operation in T-MPHN (e.g., concat, sum))",
    )
    p.add_argument(
        "--M", type=int, default=5, help="the maximum cardinality of the hypergraph"
    )
    p.add_argument(
        "--Mlst",
        type=list,
        default=[3],
        help="the maximum cardinality of the hypergraph at each layer, max(Mlst) = M",
    )
    # p.add_argument('-f') # for jupyter default
    return p.parse_args()


@dataclass
class Params:
    """
    Hyperparameter configuration for TensorHyperGNN models.

    Parameters
    ----------
    data_type : str, default="new"

    dataset : str
        Name of the dataset. Examples:
        - coauthorship: "cora", "dblp"
        - cocitation: "cora", "citeseer", "pubmed"
        - 3dObject: "ModelNet40", "NTU"
        - new: "House", "Walmart"

    model : {"T-Spectral", "T-Spatial", "T-MPHN"}
        Tensor HyperGNN model variant.

    hyperG_norm : bool, default=False
        Whether to normalize the hypergraph adjacency tensor.

    self_loop : bool, default=False
        Whether to add self-loops to the hypergraph.

    num_layers : int, default=1
        Number of HyperGNN layers.

    hid_dim : int, default=256
        Hidden-layer embedding dimension. Recommended: [64, 128, 256, 512]

    dropout : float, default=0.3
        Dropout probability applied after each HyperGNN layer.

    layernorm : bool, default=True
        Whether to apply layer normalization.

    batchnorm : bool, default=True
        Whether to apply batch normalization.

    lr : float, default=0.001
        Learning rate more sensitive since tensor operations amplifies gradients. Recommended: [0.01...0.001]
        - low lr: stable training, slower convergence, better when M is high or using deeper models
        - high lr: faster convergence, risk of divergence due to large tensor multiplication, and over generalize poorly.

    wd : float, default=5e-4
        Weight decay coefficient used in outer products and tensor multiplications which can blow up weights. Recommended: [0.005...0.0005]
        - low wd: weights grow large and modely may memorize high-order interactions
        - high wd: strong regularization and dampens high-order interactions with more stable training.

    train_ratio : float, default=0.5
        Fraction of data used for training.

    valid_ratio : float, default=0.25
        Fraction of data used for validation.

    epochs : int, default=100
        Number of training epochs.

    num_exps : int, default=2
        Number of repeated experiments.

    cuda : int, default=0
        CUDA device ID.

    seed : int, default=42
        Random seed.

    early_stopping : bool, default=True
        Whether to enable early stopping based on validation performance.

    combine : str, default="concat"
        Combination operation used in all methods when merging multiple hyperedge messages.
        Options include: "concat", "sum".
        - concat: preserves all information from each hyperedge which increase feature dimensions
        - sum: aggregates messages in lower dimension

    M : int, default=5
        defines the order of the hypergraph tensor which is the max size of any hyperedge. The model uses an M-th order adjacency tensor.
        - low M (2, 3) results in pairwise or small-group interactions: 1) faster trainig 2) better generalization for small datasets 3) may underfit if dataset has large hyperedges
        - high M (4+) results in richer joint interactions across many nodes. 1) higher expressive power 2) ideal for datasets with large hyperedges 3) computational cost grows rapidly 4) risk of overfitting 5) higher memory usage 6) training unstable if features are noisy

    Mlst : List[int], default=[3]
        defines the effects Mused at each layer of model with given method, allowing different layers to use ranging values of M to capture different representations.
        - Must satisfy: max(Mlst) == M.
        - Recommended to set orders in descending order to prevent oversmoothing.
    """

    dataset: str
    num_classes: int
    method: Literal["T-Spectral", "T-Spatial", "T-MPHN"]
    hyperG_norm: bool = False
    data_type: str = "new"
    self_loop: bool = False
    num_layers: int = 1
    hid_dim: int = 256
    dropout: float = 0.3
    layernorm: bool = True
    batchnorm: bool = True
    lr: float = 0.001
    wd: float = 5e-4
    train_ratio: float = 0.7
    valid_ratio: float = 0.10
    epochs: int = 100
    num_exps: int = 2
    device: Literal["cuda", "cpu"] = "cpu"
    seed: int = 42
    seed_weights: int = 41
    seed_splits: int = 43
    early_stopping: bool = True
    combine: str = "concat"
    M: int = 5
    Mlst: List[int] = field(default_factory=lambda: [3])
    num_features: int = None
    batch_size: int = 16


@dataclass
class TTV:
    train: Any
    test: Any
    val: Any


@dataclass
class EvalResults:
    epoch: int
    method: str
    duration: float
    loss: TTV
    accuracy: TTV
    f1: TTV
    f1_macro: TTV
    mae: TTV
    rmse: TTV
    M: int

    def __repr__(self):
        return (
            f"EvalResults("
            f"epoch={self.epoch}, \n"
            f"method='{self.method}', \n"
            f"duration={self.duration:.3f}, \n"
            f"loss={self.loss}, \n"
            f"accuracy={self.accuracy}, \n"
            f"f1={self.f1}, \n"
            f"f1_macro={self.f1_macro}, \n"
            f"mae={self.mae}, \n"
            f"rmse={self.rmse}, \n"
            f"M={self.M}"
            f")"
        )


@dataclass
class TimeTest:
    init: Any 
    model_load: Any 

    batch_size: Any 
    input_size: Any 
    target_size: Any 

    forward: Any 
    criterion: Any 
    backward: Any 
    optimizer_step: Any 

    test_eval: Any 
    val_eval: Any 


@dataclass
class Scores:
    epoch: int
    loss: float
    accuracy: float
    f1_macro: float
    f1: np.ndarray
    rmse: float

    def __repr__(self):
        return (
            f"Scores("
            f"epoch={self.epoch}, "
            f"loss={self.loss:.4f}, "
            f"accuracy={self.accuracy:.4f}, "
            f"f1_macro={self.f1_macro:.4f}, "
            f"f1={np.array2string(self.f1, precision=3)}, "
            f"rmse={self.rmse:.4f}"
            f")"
        )


def from_pygeo_to_matrices(dataset: InMemoryDataset) -> tuple:
    data, slices = dataset._data, dataset.slices
    num_subjects = dataset.y.shape[0]
    num_rois = dataset.num_features
    X_np = np.zeros((num_subjects, num_rois, num_rois), dtype=np.float32)
    A_np = np.zeros((num_subjects, num_rois, num_rois), dtype=np.float32)
    y_np = data.y.numpy()
    for i in range(num_subjects):
        x = data.x[slices["x"][i] : slices["x"][i + 1]]
        x_np = x.numpy()
        X_np[i] = x_np.reshape(num_rois, num_rois)
        edge_index = data.edge_index[
            :, slices["edge_index"][i] : slices["edge_index"][i + 1]
        ]
        A = np.zeros((num_rois, num_rois), dtype=np.float32)
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        A[src, dst] = 1
        A[dst, src] = 1  # TODO: review if I need unidirected graph for this
        A_np[i] = A
    return A_np, X_np, y_np


def from_tu_to_matrices(dataset: InMemoryDataset):
    data, slices = dataset._data, dataset.slices
    num_graphs = dataset.len()
    X_list, A_list, y_list = [], [], []
    for i in range(num_graphs):
        # Node features for graph i
        x = data.x[slices["x"][i] : slices["x"][i + 1]]
        X_np = x.numpy()  # shape (num_nodes, num_features)

        # Edge index for graph i
        edge_index = data.edge_index[
            :, slices["edge_index"][i] : slices["edge_index"][i + 1]
        ]
        num_nodes = X_np.shape[0]
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        A[src, dst] = 1
        A[dst, src] = 1  # undirected
        y = data.y[slices["y"][i] : slices["y"][i + 1]].item()

        X_list.append(X_np)
        A_list.append(A)
        y_list.append(y)
    return A_list, X_list, np.array(y_list)


class HypergraphBatchLoader:
    def __init__(
        self,
        dataset: List[Data],
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __len__(self):
        # number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            batch = [self.dataset[i] for i in batch_idx]
            yield batch
