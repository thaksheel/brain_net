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
    "./exports/rslt_a116_th.xlsx",
    "./exports/rslt_p264_th.xlsx",
    "./exports/rslt_s100_th.xlsx",
]
atlas = ["Schaefer100"]
atlas = ["AAL116", "PP264", "Schaefer100"]
atlas = ["AAL116", "Schaefer100"]
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
        M=2,
        Mlst=[2, 2],
        hid_dim=32,
        epochs=100,
        lr=5e-3,
        wd=5e-3,
        dropout=0.5,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=42,
        device="cpu",
        batch_size=128,
    )
    trainer = GraphTrainer(params, display=False, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls(dataset)
    # eval_results = trainer.evaluate_ng_cls(dataset, model_name="GATConv")
    # eval_results = trainer.evaluate_graph_cls_stnd(dataset)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=outfiles[k],
        export=True,
        exclude_fields=[],
    )
    # NOTE: running cv evals
    # seeds = [i for i in range(42, 52, 1)]
    # evals = trainer.run_stratified_n_iteractions(
    #     seeds=seeds, dataset=dataset, graph_type="stnd"
    # )
    # df = pd.DataFrame()
    # for i, seed in enumerate(seeds):
    #     df_ = trainer.evaluation_results_to_df(
    #         evaluation_results=evals[i],
    #         outname=outfiles[k],
    #         export=False,
    #     )
    #     df = pd.concat([df, df_])
    # df.to_excel(outfiles[k])

print("END")
