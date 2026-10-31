from src import (
    Affichage,
    Evaluation,
    GraphDataset,
    GridSearchParams,
    Params,
    Preprocess,
    TMPHNConfig,
)

atlas = ["Schaefer100", "AAL116", "PP264"]
rois = [100, 116, 264]
folder_name = "./data/adhd/"
methods = ["T-MPHN", "T-Spectral", "T-Spatial", "GIN", "GATConv"]
seeds = list(range(42, 42 + 30))
TUNE = False

affichage = Affichage(outdir="./exports/w44/adhd", show=False)
all_runs = []
for k, a in enumerate(atlas):
    pp = Preprocess(n_rois=rois[k], normalize="minmax")
    nodes, edges, y, errors = pp.from_node_features_to_graph_dataset(folder_name, a)
    data_list = pp.get_graph_dataset_v2(nodes, edges, y)
    dataset = GraphDataset(
        root=f"{folder_name}{a}", data_list=data_list, force_reload=False
    )
    params = Params(
        method="T-MPHN",
        dataset=a,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        M=3,
        Mlst=[3, 3],
        num_layers=2,
        hid_dim=128,
        epochs=100,
        lr=5e-4,
        wd=5e-4,
        dropout=0.65,
        train_ratio=0.6,
        valid_ratio=0.2,
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
            keep_on_device=True,
        ),
    )
    grid = GridSearchParams(
        lr=[5e-4, 1e-3, 5e-3],
        wd=[5e-4, 5e-3],
        num_layers=[1, 2],
        batch_size=[32, 64],
        dropout=[0.5, 0.65],
        hid_dim=[64, 128],
    )
    evaluation = Evaluation(
        params,
        task="graph",
        monitor="roc_auc",
        params_dir="./params/adhd",
        class_weight="balanced",
    )
    for method in methods:
        if TUNE or not evaluation.has_saved(method):
            evaluation.tune_and_evaluate(
                dataset,
                method,
                grid,
                seeds=seeds,
                maxiter=30,
                search_epochs=50,
                search_seeds=[42, 43, 44],
            )
        else:
            evaluation.evaluate_saved(dataset, method, seeds=seeds)
    affichage.report(
        evaluation.runs, name=f"adhd_{a.lower()}", searches=evaluation.searches
    )
    all_runs += evaluation.runs

print(affichage.paper_table(all_runs, "roc_auc"))
affichage.export_excel(
    "adhd_all.xlsx",
    runs=all_runs,
    sheets={
        "auc_table": affichage.paper_table(all_runs, "roc_auc"),
        "bacc_table": affichage.paper_table(all_runs, "balanced_accuracy"),
        "significance": affichage.significance_table(all_runs, "roc_auc"),
    },
)
print("END")
