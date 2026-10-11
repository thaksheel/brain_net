import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import GraphTrainer
from src.config import Params

params = Params(
    method="T-MPHN",
    dataset="HCPGender", 
    num_classes=None,
    num_features=None, 
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=16,
    epochs=200,
    lr=0.001,
    wd=0.1,
    dropout=0.8,
    train_ratio=0.7,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=8,
)

names = ["MUTAG", "NCI109", "NCI1", "PROTEINS"]
names = ["PROTEINS"]
for name in names:
    dataset = TUDataset(root="data/TUDataset", name=name)  # 4100
    params.num_features = dataset.num_features
    params.num_classes = dataset.num_classes
    params.dataset = name 

    trainer = GraphTrainer(params, display=False, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls_stnd(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=f"./exports/tu_results_{name.lower()}.xlsx",
        export=False,
        exclude_fields=[],
    )
    print(f"\n---> best_restuls: {best_results}")
print("END")
