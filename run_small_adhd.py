import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
import os
from typing import List

from src.utils.preprocess import Preprocess
from src.utils.datasets import GraphDataset
from src.config import Params
from src.train import GraphTrainer

outfiles = [
    "./exports/a116_results0.xlsx",
    "./exports/p264_results0.xlsx",
    "./exports/s100_results0.xlsx",
]
atlas = ["Schaefer100"]
atlas = ["AAL116", "PP264", "Schaefer100"]
folder_name = "./data/adhd/"
rois = [116, 264, 100]
for k, a in enumerate(atlas):
    pp = Preprocess(n_rois=rois[k], normalize="minmax")
    nodes, edges, y, errors = pp.from_node_features_to_graph_dataset(folder_name, a)
    data_list = pp.get_graph_dataset_v2(nodes, edges, y)
    dataset = GraphDataset(root=f"./data/adhd/{a}", data_list=data_list)
    params = Params(
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        method="T-MPHN",
        dataset=a,
        num_layers=2,
        M=3,
        Mlst=[3, 3],
        hid_dim=32,
        epochs=100,
        lr=5e-3,
        wd=5e-3,
        dropout=0.5,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=42,
        device="cuda",
        batch_size=256,
    )
    trainer = GraphTrainer(params, display=True, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=outfiles[k],
        export=False,
        exclude_fields=[],
    )

print("END")
