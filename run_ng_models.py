import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.train import GraphTrainer
from src.config import Params
from src.utils.datasets import NeuroGraphDataset

root_folder = "D:/datasets/hcp_data/"
dataset_name = "HCPGender"
dataset = NeuroGraphDataset(root=root_folder, name=dataset_name)
params = Params(
    num_classes=dataset.num_classes,
    num_features=dataset.num_features,
    dataset=dataset_name,
    method="T-MPHN",
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=32,
    epochs=50,
    lr=5e-3,
    wd=5e-3,
    dropout=0.65,
    train_ratio=0.6,
    valid_ratio=0.2,
    seed=42,
    device="cpu",
    batch_size=64,
)

trainer = GraphTrainer(params, display=True, progress_bar=False)
eval_results = trainer.evaluate_ng_cls(dataset, model_name="GATConv")
best_results = trainer.get_best_eval_results(eval_results)
df_results = trainer.evaluation_results_to_df(
    eval_results,
    outname="./exports/ng_results0.xlsx",
    export=False,
    exclude_fields=[],
)

print(f"\n---> best_restuls: {best_results}")
print("END")
