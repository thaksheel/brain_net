import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import GraphTrainer
from src.config import Params, GridSearchParams

names = ["MUTAG", "NCI109", "NCI1", "PROTEINS"]
params = Params(
    method="T-MPHN",
    dataset=None,
    num_classes=None,
    num_features=None,
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
    lr=[5e-3, 5e-2, 5e-1, 1e-3, 1e-2, 1e-1],
    wd=[5e-3, 5e-2, 5e-1, 1e-3, 1e-2, 1e-1],
    num_layers=[2],
    batch_size=[4, 8, 16, 32, 64, 128],
    dropout=[0.25, 0.5, 0.75, 0.8],
    hid_dim=[16, 32, 64, 128],
)

for name in names:
    dataset = TUDataset(root="data/TUDataset", name=name)  # 4100
    params.num_features = dataset.num_features
    params.num_classes = dataset.num_classes
    params.dataset = name
    trainer = GraphTrainer(params, display=False, progress_bar=False)
    best_params, best_score = trainer.grid_search(
        dataset,
        gsp,
        graph_type="stnd",
        maxiter=100,
    )
    updated_params = trainer.set_params(**best_params)
    trainer.params = updated_params
    trainer.progress_bar = True
    eval_results = trainer.evaluate_graph_cls_stnd(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=f"./exports/tu_results_{name.lower()}.xlsx",
        export=True,
    )
    print(f"\n---> best_restuls: {best_results}")
print("END")
