from src import Affichage, Evaluation, NodeTrainer, Params, TMPHNConfig

folder_name = "./data/academic_net/"
datasets = ["cora-cc", "cora-ca", "dblp-ca", "citeseer-cc"]
methods = ["T-MPHN", "MLP"]
seeds = list(range(42, 42 + 10))

affichage = Affichage(outdir="./exports/w44/node", show=False)
all_runs = []
for d in datasets:
    params = Params(
        method="T-MPHN",
        dataset=d,
        num_features=0,
        num_classes=0,
        Mlst=[5, 5],
        num_layers=2,
        hid_dim=128,
        epochs=100,
        lr=1e-3,
        wd=1e-3,
        dropout=0.2,
        train_ratio=0.5,
        valid_ratio=0.25,
        seed=42,
        device="cuda",
        tmphn_config=TMPHNConfig(
            norm="rms",
            project_input=True,
            strategy="random",
            checkpoint=False,
            chunk_size=None,
        ),
    )
    data = NodeTrainer(params).load_dataset(folder_name + d)
    evaluation = Evaluation(
        params, task="node", monitor="accuracy", params_dir="./params/node"
    )
    all_runs += evaluation.compare(data, methods, seeds=seeds)

affichage.report(all_runs, name="node_cls", metric="accuracy")
print(affichage.paper_table(all_runs, "accuracy", percent=True))
print("END")
