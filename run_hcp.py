from src import (
    Affichage,
    Evaluation,
    GridSearchParams,
    NeuroGraphDataset,
    Params,
    TMPHNConfig,
)

ROOT = "C:/Users/tnall/Downloads/"
ROOT = "D:/datasets/hcp_data/"
names = ["HCPGender", "HCPAge", "HCPTask"]
methods = ["T-MPHN", "GATConv", "GIN"]
seeds = list(range(42, 42 + 30))
TUNE = False

affichage = Affichage(outdir="./exports/w44/hcp", show=False)
all_runs = []
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
    grid = GridSearchParams(
        lr=[5e-4, 1e-3, 5e-3, 1e-2],
        wd=[5e-4, 5e-3, 5e-2],
        num_layers=[2, 3],
        batch_size=[32, 64],
        dropout=[0.5, 0.65, 0.75],
        hid_dim=[64, 128],
    )
    evaluation = Evaluation(
        params, task="graph", monitor="roc_auc", params_dir="./params/hcp"
    )
    for method in methods:
        if TUNE or not evaluation.has_saved(method):
            evaluation.tune_and_evaluate(
                dataset, method, grid, seeds=seeds, maxiter=30, search_epochs=50
            )
        else:
            evaluation.evaluate_saved(dataset, method, seeds=seeds)
    affichage.report(
        evaluation.runs, name=f"hcp_{name.lower()}", searches=evaluation.searches
    )
    all_runs += evaluation.runs

print(affichage.summary_text(all_runs, metrics=("accuracy", "roc_auc", "f1")))
affichage.export_excel(
    "hcp_all.xlsx",
    runs=all_runs,
    sheets={
        "accuracy_table": affichage.paper_table(all_runs, "accuracy", percent=True),
        "auc_table": affichage.paper_table(all_runs, "roc_auc"),
        "significance": affichage.significance_table(all_runs, "roc_auc"),
    },
)
print("END")
