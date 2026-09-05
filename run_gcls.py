import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import THGTrainer
from src.config import Params


params = Params(
    method="T-MPHN",
    dataset="HCPGender",
    num_classes=None,
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=64,
    epochs=200,
    lr=5e-3,
    wd=5e-3,
    dropout=0.65,
    train_ratio=0.7,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=32,
)

dataset = TUDataset(root="data/TUDataset", name="NCI109")  # 4100
dataset = TUDataset(root="data/TUDataset", name="MUTAG")  # 186
dataset = TUDataset(root="data/TUDataset", name="PROTEINS")  # 1100
dataset = TUDataset(root="data/TUDataset", name="NCI1")  # 4100
params.num_features = dataset.num_features
params.num_classes = dataset.num_classes

trainer = THGTrainer(params, display=True)
eval_results = trainer.evaluate_graph_cls(dataset)
best_results = trainer.get_best_eval_results(eval_results)
df_results = trainer.evaluation_results_to_df(
    eval_results,
    outname="./exports/tu_results5.xlsx",
    export=True,
    exclude_fields=[],
)
print(f"\n---> best_restuls: {best_results}")
print("END")
