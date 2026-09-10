import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Literal
from torch_geometric.data import Data
import pickle


@dataclass
class NeuroGraphParams:
    root_folder: str
    dataset_name: Literal["HCPGender", "HCPTask", "HCPAge"] = (
        "HCPGender"  # TODO:add more dataset
    )
    runs: int = 1
    device: torch.device = torch.device("cuda")
    seed: int = 123
    model: Literal[
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
    ] = "GCNConv"
    hidden: int = 32
    hidden_mlp: int = 64
    num_layers: int = 3
    epochs: int = 100
    echo_epoch: int = 50
    batch_size: int = 16
    early_stopping: int = 50
    lr: float = 1e-5
    wd: float = 1e-4
    dropout: float = 0.5
    num_features: int = None
    num_classes: int = None


@dataclass
class TTV:
    train: Any
    test: Any
    val: Any


@dataclass
class EvalResults:
    epoch: int
    method: str
    loss: TTV
    rmse: TTV
    mae: TTV
    r2: TTV
    accuracy: TTV
    f1: TTV
    duration: float
    M: int
    model: torch.nn.Module


def from_pygeo_to_matrices(filepath: str, outpath: str) -> Dict[str, np.ndarray]:
    data: Data
    slices: Dict
    data, slices = torch.load(filepath, weights_only=False)
    num_subjects = slices["x"].size(0) - 1
    num_rois = data.num_features
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
    D = {
        "A": A_np,
        "X": X_np,
        "Y": y_np,
    }
    with open(outpath, "wb") as f:
        pickle.dump(D, f)
    return D
