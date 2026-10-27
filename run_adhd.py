import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import (
    GraphTrainer,
    Preprocess,
    GraphDataset,
    Params,
    TMPHNConfig,
    GridSearchParams,
)

outfiles = [
    "./exports/w43/adhd_rslt_s100.xlsx",
    "./exports/w43/adhd_rslt_a116.xlsx",
    "./exports/w43/adhd_rslt_p264.xlsx",
]
outfiles = [
    "./exports/w43/adhd_rslt_a116.xlsx",
    "./exports/w43/adhd_rslt_p264.xlsx",
]
atlas = ["Schaefer100", "AAL116", "PP264"]
atlas = ["AAL116", "PP264"]
folder_name = "./data/adhd/"
rois = [100, 116, 264]
rois = [116, 264]
for k, a in enumerate(atlas):
    pp = Preprocess(n_rois=rois[k], normalize="minmax")
    nodes, edges, y, errors = pp.from_node_features_to_graph_dataset(folder_name, a)
    data_list = pp.get_graph_dataset_v2(nodes, edges, y)
    dataset = GraphDataset(
        root=f"./data/adhd/{a}", data_list=data_list, force_reload=True
    )
    params = Params(
        method="T-Spectral",
        dataset=atlas,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        M=3,
        Mlst=[3, 3],
        num_layers=2,
        hid_dim=128,
        epochs=400,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=42,
        device="cuda",
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
    trainer = GraphTrainer(params, display=True, progress_bar=True)
    # eval_results = trainer.evaluate_graph_cls(dataset) # TMPHN
    eval_results = trainer.evaluate_graph_cls_spectral(dataset)
    best_results = trainer.get_best_eval_results(eval_results)
    print(f"best_results={best_results}")

    # reporting
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=outfiles[k],
        export=True,
        exclude_fields=[],
    )

print("END")
