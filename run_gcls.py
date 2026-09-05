import torch
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch_geometric.datasets import TUDataset

from src.train import THGTrainer
from src.config import Params


def run_single_dataset(name: str):
    print(f"\n\n--->[START] {name}")
    # Create a fresh Params object INSIDE the thread otherwise error
    params = Params(
        method="T-MPHN",
        dataset=name,
        num_classes=None,
        num_layers=2,
        M=2,
        Mlst=[2, 2],
        hid_dim=64,
        epochs=1,
        lr=5e-3,
        wd=5e-3,
        dropout=0.65,
        train_ratio=0.7,
        valid_ratio=0.1,
        seed=42,
        device="cpu",
        batch_size=32,
    )
    dataset = TUDataset(root="data/TUDataset", name=name)
    params.num_features = dataset.num_features
    params.num_classes = dataset.num_classes

    trainer = THGTrainer(params, display=True)
    eval_results = trainer.evaluate_graph_cls(dataset)
    best_results = trainer.get_best_eval_results(eval_results)

    print(f"\n ---> [DONE] {name} → best_results: {best_results}")
    return name, eval_results, best_results


names = ["PROTEINS", "NCI109", "NCI1", "MUTAG"]
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_single_dataset, name): name for name in names}
    for future in as_completed(futures):
        name = futures[future]
        ds_name, eval_results, best_results = future.result()
        df_results = THGTrainer(Params(dataset=None, num_classes=None, method=None)).evaluation_results_to_df(
            eval_results,
            outname=f"./exports/tu_results_{name.lower()}.xlsx",
            export=True,
            exclude_fields=[],
        )
        print(f"\n---> best_results ({name}): {best_results}")

print("END")
