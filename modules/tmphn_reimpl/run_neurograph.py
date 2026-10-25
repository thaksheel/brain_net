"""
T-MPHN graph classification on the NeuroGraph HCP datasets:
metrics and runtime for every epoch, one Excel file (+ learning-curve PNG) per dataset.

Put this file, graph_cls_utils.py and the `tmphn/` folder in your project root (next to `src/`):
    python run_neurograph.py
"""
from graph_cls_utils import Config, is_classification, run_experiment
from datasets import NeuroGraphDataset

ROOT = "D:/datasets/hcp_data/"
NAMES = ["HCPTask"]  # HCPFI / HCPWM have continuous targets and are skipped
NAMES = ["HCPGender", "HCPAge", "HCPTask"]  # HCPFI / HCPWM have continuous targets and are skipped

cfg = Config(
    Mlst=[3, 3],
    hid_dim=128,
    epochs=400,               # your script overrode epochs=100 with trainer.params.epochs = 50
    lr=5e-4,                 # your value (very low for 50 epochs; 1e-3 is a more typical start)
    wd=5e-4,
    dropout=0.5,
    batch_size=64,
    train_ratio=0.7,
    valid_ratio=0.1,
    hyperedges="neighborhood",  # each ROI + its connected ROIs -> real 3-node interactions at M = 3
    strategy="topk",            # keep the most strongly connected members (|FC|) when down-sampling to M
    checkpoint=True,            # ~1000-ROI subjects: recompute products in backward to save GPU memory
    chunk_size=2_000_000,
    keep_on_device=False,       # too large to keep every subject on the GPU; batches are moved instead
    seed=42,
)

if __name__ == "__main__":
    for name in NAMES:
        dataset = NeuroGraphDataset(root=ROOT, name=name)
        if not is_classification(dataset):
            print(f"\nskipping {name}: targets are continuous (regression), not class labels")
            continue
        run_experiment(dataset, name, cfg, outfile=f"./exports/tmphn_epochs_{name.lower()}.xlsx")
    print("END")
