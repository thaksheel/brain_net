import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset

from src.train import GraphTrainer
from src.config import Params
from src.utils.datasets import NeuroGraphDataset
from src.utils.preprocess import Preprocess
from src.utils.datasets import GraphDataset


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
        epochs=1,
        lr=5e-3,
        wd=5e-3,
        dropout=0.5,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=42,
        device="cpu",
        batch_size=128,
    )
    trainer = GraphTrainer(params, display=True, progress_bar=False)
    eval_results = trainer.evaluate_graph_cls(dataset)

root_folder = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPTask"]
# for name in names:
#     dataset = NeuroGraphDataset(root=root_folder, name=name)
#     params = Params(
#         num_classes=dataset.num_classes,
#         num_features=dataset.num_features,
#         dataset=name,
#         method="T-MPHN",
#         num_layers=2,
#         M=3,
#         Mlst=[3, 3],
#         hid_dim=32,
#         epochs=1,
#         lr=5e-3,
#         wd=5e-3,
#         dropout=0.65,
#         train_ratio=0.6,
#         valid_ratio=0.2,
#         seed=42,
#         device="cpu",
#         batch_size=64,
#     )
#     trainer = GraphTrainer(params, display=True, progress_bar=False)
#     eval_results = trainer.evaluate_graph_cls(dataset)

tu_names = ["MUTAG", "NCI109", "NCI1", "PROTEINS"]
for name in tu_names:
    dataset = TUDataset(root="data/TUDataset", name=name) 
    params = Params(
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        dataset=name,
        method="T-MPHN",
        num_layers=2,
        M=3,
        Mlst=[3, 3],
        hid_dim=32,
        epochs=1,
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
    eval_results = trainer.evaluate_graph_cls(dataset)


print("END")
