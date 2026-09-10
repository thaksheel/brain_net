import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import GraphTrainer
from src.config import Params

params = Params(
    method="T-MPHN",
    dataset=None,
    num_classes=None,
    num_features=None,
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=64,
    epochs=100,
    lr=5e-3,
    wd=5e-3,
    dropout=0.65,
    train_ratio=0.7,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=32,
)

dataset = TUDataset(root="data/TUDataset", name="MUTAG")
params.num_features = dataset.num_features
params.num_classes = dataset.num_classes
params.dataset = "MUTAG"

trainer = GraphTrainer(params, display=True)
seeds = [i for i in range(42, 52, 1)]
evals = trainer.run_stratified_n_iteractions(
    seeds=seeds, dataset=dataset, graph_type="stnd"
)
df = pd.DataFrame()
for i, seed in enumerate(seeds):
    df_ = trainer.evaluation_results_to_df(
        evaluation_results=evals[i],
        outname="./exports/results_n_iters.xlsx",
        export=False,
    )
    df = pd.concat([df, df_])

print(len(df))
print("END")
