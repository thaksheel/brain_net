import pickle

from src.train import GraphTrainer
from src.config import Params


with open("./data/A_gender_static.pkl", "rb") as f:
    H = pickle.load(f)
with open("./data/X_gender_static.pkl", "rb") as f:
    X = pickle.load(f)
with open("./data/y_gender_static.pkl", "rb") as f:
    y = pickle.load(f)


params = Params(
    method="T-MPHN",
    dataset="HCPGender",
    num_classes=2,
    num_features=X.shape[1],
    num_layers=2,
    M=7,
    Mlst=[7, 7],
    hid_dim=16,
    epochs=200,
    lr=0.001,
    wd=0.1,
    dropout=0.8,
    train_ratio=0.7,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=8,
)
gt = GraphTrainer(
    params=params,
    init_limit_rois=10,
    limit_X_features=20,
    init_stopper=30,
    display=False,
)
train_results = gt.train_node_cls(X[:50, :50 , 0], H[:50, :50 , 0], y[:50])
print("END")
