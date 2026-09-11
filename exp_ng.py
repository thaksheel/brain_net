import pandas as pd

from src.train import GraphTrainer
from src.config import Params, GridSearchParams
from src.utils.datasets import NeuroGraphDataset

root_folder = "C:/Users/tnall/Downloads/"
root_folder = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPFI", "HCPTask", "HCPWM"]
names = ["HCPGender", "HCPAge"]
names = ["HCPTask", "HCPWM"]
for name in names:
    dataset = NeuroGraphDataset(root=root_folder, name=name)
    params = Params(
        num_classes=dataset.num_classes,
        num_features=dataset.num_features,
        dataset=name,
        method="T-MPHN",
        num_layers=2,
        M=3,
        Mlst=[3, 3],
        hid_dim=32,
        epochs=50,
        lr=5e-3,
        wd=5e-3,
        dropout=0.65,
        train_ratio=0.6,
        valid_ratio=0.2,
        seed=42,
        device="cpu",
        batch_size=64,
    )
    gsp = GridSearchParams(
        lr=[5e-3, 5e-2, 5e-1, 1e-3, 1e-2, 1e-1],
        wd=[5e-3, 5e-2, 5e-1, 1e-3, 1e-2, 1e-1],
        num_layers=[2],
        batch_size=[17, 32, 64, 128],
        dropout=[0.25, 0.5, 0.75, 0.8],
        hid_dim=[16, 32, 64, 128],
    )
    trainer = GraphTrainer(params, display=False, progress_bar=True)

    # NOTE: params tuning
    trainer.params.epochs = 15
    best_params, best_score = trainer.grid_search(
        dataset, gsp, graph_type="ng", maxiter=50, model_name="GATConv"
    )
    updated_params = trainer.set_params(**best_params)
    print(f"grid_search results: {updated_params}")

    # NOTE: CV evaluations
    trainer.params.epochs = 50
    seeds = [i for i in range(42, 42 + 30, 1)]
    evals = trainer.run_stratified_n_iteractions(
        seeds=seeds, dataset=dataset, graph_type="ng", model_name="GATConv"
    )
    df = pd.DataFrame()
    for i, seed in enumerate(seeds):
        df_ = trainer.evaluation_results_to_df(
            evaluation_results=evals[i],
            outname=None,
            export=False,
        )
        df = pd.concat([df, df_])
    df.to_excel(f"./exports/rslt_{name.lower()}_exp_ng.xlsx")

print("END")
