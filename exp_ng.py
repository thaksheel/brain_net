import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import Params, GraphTrainer, TMPHNConfig, NeuroGraphDataset, GridSearchParams

ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender"]
names = ["HCPGender", "HCPAge", "HCPTask"]

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
        epochs=250,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.7,
        valid_ratio=0.1,
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
    gsp = GridSearchParams(
        lr=[5e-4, 5e-3, 5e-2, 5e-1, 1e-3, 1e-2],
        wd=[5e-4, 5e-3, 5e-2, 5e-1, 1e-3, 1e-2],
        num_layers=[3],
        batch_size=[16, 32, 64],
        dropout=[0.5, 0.65, 0.75],
        hid_dim=[32, 64, 128],
    )

    # # NOTE: params tuning
    trainer.params.epochs = 75
    best_params, best_score = trainer.grid_search(
        dataset,
        gsp,
        graph_type="tensor",
        maxiter=100,
    )
    updated_params = trainer.set_params(**best_params)
    print(f"grid_search results: {updated_params}")

    # NOTE: CV evaluations
    trainer = GraphTrainer(params, display=False, progress_bar=True)
    trainer.params.epochs = 250
    seeds = [i for i in range(42, 42 + 30, 1)]
    evals = trainer.run_stratified_n_iteractions(
        seeds=seeds,
        dataset=dataset,
        graph_type="tensor",
    )
    df = pd.DataFrame()
    for i, seed in enumerate(seeds):
        df_ = trainer.evaluation_results_to_df(
            evaluation_results=evals[i],
            outname=None,
            export=False,
        )
        df = pd.concat([df, df_])
    df.to_excel(f"./exports/rslt_{name.lower()}.xlsx")

print("END")
