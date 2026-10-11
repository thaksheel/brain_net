"""
T-MPHN graph classification on the ADHD atlases (replaces run_graph_classification.py):
metrics and runtime for every epoch, one Excel file (+ learning-curve PNG) per atlas.

Put this file, graph_cls_utils.py and the `tmphn/` folder in your project root (next to `src/`):
    python run_adhd.py
"""
from graph_cls_utils import Config, run_experiment
from src import GraphDataset, Preprocess

DATA_DIR = "./data/adhd/"
ATLAS = ["AAL116", "PP264", "Schaefer100"]
ROIS = [116, 264, 100]
OUTFILES = [
    "./exports/tmphn_epochs_a116.xlsx",
    "./exports/tmphn_epochs_p264.xlsx",
    "./exports/tmphn_epochs_s100.xlsx",
]

cfg = Config(
    Mlst=[2, 2],          # M = 2: plain mean-of-neighbors GNN (try "neighborhood" + [3, 3] for 3-node interactions)
    hid_dim=32,
    epochs=100,
    lr=5e-3,
    wd=5e-3,
    dropout=0.5,
    batch_size=128,
    train_ratio=0.6,
    valid_ratio=0.2,
    hyperedges="edges",
    keep_on_device=True,  # small dataset: move every subject to the GPU once
    seed=42,
)

if __name__ == "__main__":
    print(cfg)
    for k, atlas in enumerate(ATLAS):
        pp = Preprocess(n_rois=ROIS[k], normalize="minmax")
        nodes, edges, y, errors = pp.from_node_features_to_graph_dataset(DATA_DIR, atlas)
        data_list = pp.get_graph_dataset_v2(nodes, edges, y)
        dataset = GraphDataset(root=f"{DATA_DIR}{atlas}", data_list=data_list)
        run_experiment(dataset, atlas, cfg, outfile=OUTFILES[k])
    print("END")
