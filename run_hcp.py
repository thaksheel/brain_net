import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import Params, GraphTrainer, TMPHNConfig, NeuroGraphDataset

ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPTask"]
names = ["HCPTask"]

for name in names:
    dataset = NeuroGraphDataset(root=ROOT, name=name)
    params = Params(
        method="T-MPHN",
        dataset=name,
        num_features=dataset.num_features, 
        num_classes=dataset.num_classes,
        M=3,
        Mlst=[3, 3],
        num_layers=2, 
        hid_dim=128,
        epochs=100,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.7,
        valid_ratio=0.1,
        seed=42,
        device="cpu",
        batch_size=64,
        tmphn_config=TMPHNConfig(
            pool="mean",
            norm="rms",
            project_input=True,
            hyperedges="neighborhood",
            strategy="topk",
            checkpoint=True,
            chunk_size=2_000_000,
            keep_on_device=False,
        ),
    )

    trainer = GraphTrainer(params=params, display=True, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=f"./exports/w43/ng_rslt_{name.lower()}.xlsx",
        export=True,
        exclude_fields=[],
    )

print("END")
