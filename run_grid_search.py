import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import GraphTrainer
from src.config import Params, GridSearchParams

name = "MUTAG"
dataset = TUDataset(root="data/TUDataset", name=name)
params = Params(
    method="T-MPHN",
    dataset=name,
    num_classes=dataset.num_classes,
    num_features=dataset.num_features,
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=64,
    epochs=200,
    lr=5e-3,
    wd=5e-3,
    dropout=0.65,
    train_ratio=0.6,
    valid_ratio=0.2,
    seed=42,
    device="cpu",
    batch_size=32,
)
gsp = GridSearchParams(
    lr=[1e-3, 1e-2, 1e-1],
    wd=[1e-3, 1e-2, 1e-1],
    num_layers=[2],
    batch_size=[8, 32, 64],
    dropout=[0.2, 0.5, 0.8],
    hid_dim=[16, 64, 128],
)

trainer = GraphTrainer(params, display=False)
best_params, best_score = trainer.grid_search(
    dataset,
    gsp,
    graph_type="stnd",
    maxiter=100,
)
updated_params = trainer.set_params(**best_params)
print(f"\n\nbest_params={best_params}")
print(f"best_score={best_score}")
print("END")
