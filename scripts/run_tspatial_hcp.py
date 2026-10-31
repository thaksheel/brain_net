import torch

from src import Params, GraphTrainer, TMPHNConfig, NeuroGraphDataset

ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPTask"]
for name in names:
    dataset = NeuroGraphDataset(root=ROOT, name=name)
    num_nodes = dataset[0].num_nodes
    params = Params(
        method="T-Spatial",
        dataset=name,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        M=3,
        Mlst=[3, 3],
        num_layers=2,
        hid_dim=32,
        epochs=100,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.7,
        valid_ratio=0.1,
        seed=42,
        device="cuda",
        batch_size=2,
        self_loop=True,            # the paper's T-spatial: self-loop adjacency + summation
        tmphn_config=TMPHNConfig(
            pool="mean",
            norm="rms",
            project_input=True,
            hyperedges="neighborhood",
            strategy="topk",
        ),
    )
    params.tmphn_config.spatial_combine = "sum"   # "sum" | "concat" (= T-MPHN) | "none" (original layer)

    trainer = GraphTrainer(params=params, display=False, progress_bar=True)
    eval_results = trainer.evaluate_graph_cls_spatial(dataset)

    # reporting 
    print(trainer.get_best_eval_results(eval_results))
    trainer.evaluation_results_to_df(
        eval_results,
        outname=f"./exports/w43/ng_tspatial_rslt_{name.lower()}.xlsx",
        export=True,
        exclude_fields=[],
    )
    torch.cuda.empty_cache()

print("END")
