from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import torch
import time

from src.config import Params
from src.models.tmphn_graph import TMPHN_Graph_Cls


def evaluate(model, loader, params, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            y = batch.y.view(-1)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.size(0)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = correct / total
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2))
    return acc, total_loss, rmse


tu_dataset = TUDataset(root="data/TUDataset", name="MUTAG") # 186
tu_dataset = TUDataset(root="data/TUDataset", name="NCI1") #4100
tu_dataset = TUDataset(root="data/TUDataset", name="NCI109") #4100
tu_dataset = TUDataset(root="data/TUDataset", name="PROTEINS") #1100
split = 750
params = Params(
    num_classes=tu_dataset.num_classes,
    method="T-MPHN",
    dataset="HCPGender",
    num_layers=2,
    M=3,
    Mlst=[3, 3],
    hid_dim=64,
    epochs=100,
    lr=5e-3,
    wd=5e-3,
    dropout=0.25, 
    train_ratio=0.6,
    valid_ratio=0.1,
    seed=42,
    device="cpu",
    batch_size=1,
)
train_loader = DataLoader(tu_dataset[:split], batch_size=32, shuffle=True)
test_loader = DataLoader(tu_dataset[split:], batch_size=32, shuffle=True)
params.num_features = tu_dataset.num_features
model = TMPHN_Graph_Cls(params).to(torch.device(params.device))
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=params.lr,
    weight_decay=params.wd,
)
criterion = torch.nn.CrossEntropyLoss()
results = []
for epoch in range(params.epochs):
    model.train()
    epoch_loss = 0
    start = time.time()
    for batch in train_loader:
        out = model.forward(batch)
        # y = torch.cat([d.y for d in batch]).view(-1)
        y = batch.y.view(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    runtime = time.time() - start
    test_acc, test_loss, test_rmse = evaluate(model, test_loader, params, criterion)
    train_preds = out.argmax(dim=1).cpu().numpy()
    train_trues = y.cpu().numpy()
    train_rmse = np.sqrt(np.mean((train_preds - train_trues) ** 2))
    results.append(
        {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "test_loss": test_loss,
            "train_acc": (train_preds == train_trues).mean(),
            "test_acc": test_acc,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "runtime": runtime,
        }
    )
    df = pd.DataFrame(results)
    df.to_excel("./exports/tu_results_pro0.xlsx", index=False)
    print(
        f"Epoch {epoch+1:03d} "
        f"Train Loss: {epoch_loss:.4f} "
        f"Test Acc: {test_acc:.4f} "
        f"Runtime : {runtime:.2f}s"
    )

df = pd.DataFrame(results)
print(df)
df.to_excel("./exports/tu_results_pro0.xlsx", index=False)

print("END")
# Runtime ~ 23 mins @ 500 epochs
