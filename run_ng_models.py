import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.neurograph import *

params = NeuroGraphParams(
    root_folder="C:/Users/tnall/Downloads/",
    # root_folder="D:/hcp-data/",
    # root_folder="E:/hcp-data/",
    dataset_name="HCPGender",
    device=torch.device("cpu"),
    model="GraphConv",
    epochs=100,
    batch_size=64,
    lr=1e-5,
    wd=1e-4,
)
neurograph = NGEstimator(params=params, display=True)
ng_model, evaluation_results = neurograph.fit_train()
df = neurograph.evaluation_results_to_df(
    evaluation_results,
    outname=None,
    export=False,
    exclude_fields=["model", "M"],
)
df.to_excel("./exports/results_gender.xlsx")
best_results = neurograph.get_best_eval_results(evaluation_results)

print(
    f"best_results >> "
    f"epoch={best_results.epoch} "
    f"test_acc={100*best_results.accuracy.test:2f}% "
    f"test_f1={100*best_results.f1.test:2f}% "
    f"total_runtime={np.sum([er.duration for er in evaluation_results]):2f}s"
)
print(df.head())
print("END")
