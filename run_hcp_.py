from src import Affichage, GraphTrainer, NeuroGraphDataset, Params, TMPHNConfig

ROOT = "./data/"
ROOT = "C:/Users/tnall/Downloads/"
ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender"]
methods = ["T-MPHN"]

affichage = Affichage(outdir="./exports/w44/hcp", show=False)
runs = []
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
        hid_dim=64,
        epochs=50,
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
    trainer = GraphTrainer(params, display=False, progress_bar=True, monitor="roc_auc")
    for method in methods:
        run = trainer.train(dataset, method=method)
        print(run)
        runs.append(run)

print(
    affichage.summary_text(
        runs, metrics=("accuracy", "balanced_accuracy", "roc_auc", "f1")
    )
)
affichage.report(runs, name="hcp_single")
print("END")
