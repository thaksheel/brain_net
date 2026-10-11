import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import GraphTrainer, Params, EvalResults 


def load_dataset(path: str):
    with open(path + "/hypergraph.pickle", "rb") as f:
        H = pickle.load(f)
    with open(path + "/feature.pickle", "rb") as f:
        X = pickle.load(f)
    with open(path + "/labels.pickle", "rb") as f:
        y = pickle.load(f)
    return X, H, y


params = Params(
    method="T-Spatial",
    dataset="HCPGender",
    num_classes=2,
    num_features=None,
    num_layers=2,
    M=2,
    Mlst=[2, 2],
    hid_dim=128,
    epochs=200,
    lr=0.001,
    wd=0.001,
    dropout=0.65,
    train_ratio=0.5,
    valid_ratio=0.25,
    seed=42,
    device="cpu", # TODO: cuda does not work right now, error 
    batch_size=32,
)

datasets = ["cora-cc", "cora-ca", "dblp-ca", "citeseer-cc"]
seeds = [s for s in range(42, 42+15, 1)]
for d in datasets:
    best_results: list[EvalResults] = []
    for seed in seeds:
        folder_path = "./data/academic_net/"
        X, H, y = load_dataset(path=folder_path + d)
        params.num_features = X.shape[1]
        params.seed = seed
        gt = GraphTrainer(params=params, display=True)
        train_results = gt.train_node_cls(X, H, y)
        be = gt.get_best_eval_results(train_results)
        best_results.append(be)
    df = gt.evaluation_results_to_df(best_results, outname=f"./exports/rslt_{d}.xlsx", export=True)
    print(df.head())
print("END")
