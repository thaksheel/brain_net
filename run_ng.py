import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import THGTrainer
from src.config import Params
from src.utils.datasets import NeuroGraphDataset

params = Params(
    method="T-MPHN",
    dataset="HCPGender",
    num_classes=None,
    num_layers=2,
    M=2,
    Mlst=[2, 2],
    hid_dim=32,
    epochs=200,
    lr=5e-3,
    wd=5e-3,
    dropout=0.65,
    train_ratio=0.7,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=16,
)
root_folder="D:/datasets/hcp_data/"
dataset_name="HCPGender"
dataset = NeuroGraphDataset(root=root_folder, name=dataset_name)
params.num_features = dataset.num_features
params.num_classes = dataset.num_classes

trainer = THGTrainer(params, display=True)
eval_results = trainer.evaluate_graph_cls(dataset)
best_results = trainer.get_best_eval_results(eval_results)
df_results = trainer.evaluation_results_to_df(
    eval_results,
    outname="./exports/ng_results0.xlsx",
    export=True,
    exclude_fields=[],
)
print(f"\n---> best_restuls: {best_results}")
print("END")
