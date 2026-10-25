import torch

from src import Params, GraphTrainer, TMPHNConfig, NeuroGraphDataset

ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPTask"]

for name in names:
    dataset = NeuroGraphDataset(root=ROOT, name=name)
    num_nodes = dataset[0].num_nodes
    params = Params(
        method="T-Spectral",
        dataset=name,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        M=3,
        Mlst=[3],  # T-spectral is fixed at M = 3; kept for EvalResults/logging
        num_layers=1,  # the paper uses one T-spectral layer (it is already global)
        hid_dim=64,
        epochs=100,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.7,
        valid_ratio=0.1,
        seed=42,
        device="cuda",
        batch_size=2,  # each 1000-ROI subject needs ~4 GB for its adjacency spectrum
        tmphn_config=TMPHNConfig(
            pool="mean",
            norm="rms",
            project_input=True,
            hyperedges="neighborhood",
            strategy="topk",
        ),
    )
    params.tmphn_config.readout = (
        "learned"  # "learned" | "first" (= T-spatial) | "sum" (original)
    )
    trainer = GraphTrainer(params=params, display=True, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls_spectral(dataset)

    # reporting 
    best_results = trainer.get_best_eval_results(eval_results)
    print(best_results)
    df_results = trainer.evaluation_results_to_df(
        eval_results,
        outname=f"./exports/w43/ng_tspec_rslt_{name.lower()}.xlsx",
        export=True,
        exclude_fields=[],
    )
    torch.cuda.empty_cache()

print("END")
