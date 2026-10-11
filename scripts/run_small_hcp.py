import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.utils.preprocess import Preprocess
from src.utils.datasets import GraphDataset
from src.config import Params
from src.train import GraphTrainer

atlas = ["AAL116", "PP264", "Schaefer100"]
outfiles = [
    "./exports/a116_results2.xlsx",
    "./exports/p264_results2.xlsx",
    "./exports/s100_results2.xlsx",
]
rois = [116, 264, 100]
for k, a in enumerate(atlas):
    pp = Preprocess(n_rois=rois[k], normalize="minmax")
    y, nodes, edges = pp.read_data(
        target_path="./data/targets.csv",
        node_folder=f"./data/node/{a}",
        edge_folder=f"./data/edge/{a}",
        node_rep="mean",
        edge_rep="top_frequency",
        sparsity="0.5",
    )
    data_list = pp.get_graph_dataset(nodes, edges, y)
    dataset = GraphDataset(root="./data/", data_list=data_list)
    params = Params(
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        method="T-MPHN",
        dataset=a,
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
    trainer = GraphTrainer(params, display=True, collect_time_test=True)
    eval_results = trainer.evaluate_graph_cls(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname="./exports/ng_small_results0.xlsx",
        export=False,
        exclude_fields=[],
    )

print("END")
