# Tensorized Hypergraph GNN for Brain Network Classification

This repository contains a modified implementation of Tensorized Hypergraph Neural Networks (T-HyperGNNs) for brain network analysis.

The original models were designed for node classification. This version reimplements all three models, extends them to graph classification (one brain network per subject), and adds an evaluation pipeline for repeated, reproducible experiments on brain network datasets.

## Project Status

This project is an active research implementation.

- All three Tensorized HyperGNN models (T-MPHN, T-Spectral, T-Spatial) now train correctly for both graph classification and node classification.
- The pipeline runs on the **ADHD dataset** (Schaefer100, AAL116 and PP264 atlases) and on the **NeuroGraph HCP datasets** (HCPGender, HCPAge, HCPTask).
- NeuroGraph baselines (`ResidualGNNs` with GCN, GAT, SAGE, ... layers) and a GIN baseline are evaluated with the same splits and metrics, so results are directly comparable.
- The evaluation pipeline supports grid search, repeated stratified splits, k-fold cross-validation, saved parameter sets, summary tables (mean ± std) and significance tests.


## Tensorized HyperGNN Models

| Model | Idea | Memory | Use it for |
|---|---|---|---|
| **T-MPHN** | Message passing over compiled hypergraph terms | Linear in the number of nodes | Any dataset, including 1000-ROI HCP |
| **T-Spectral** | t-product in the spectral domain with a learnable frequency filter | O(N³) per subject | Atlases with about 100 to 300 ROIs (ADHD) |
| **T-Spatial** | First slice of the t-product (dense adjacency tensor) | O(N³) per subject | Atlases with about 100 to 300 ROIs (ADHD) |

### Changes from the original implementation

- **Reimplemented T-MPHN:** each layer runs once for all nodes, and there is no recursion. The hypergraph is compiled once, so the model runs on the GPU and converges.
- **Fixed T-Spectral and T-Spatial:** these previously diverged or did not update. The fixes are symmetrization of the t-product, real-valued spectra, normalization, self terms and biases.
- **Hyperedges built from brain graphs:** `neighborhood` (each ROI with its connected ROIs), `edges`, or `knn`. Oversized hyperedges can be down-sampled to the most strongly connected ROIs (`strategy="topk"`, using |FC|).

## Quick Start

```python
from src import Affichage, Evaluation, GraphTrainer

# one run of one method
run = GraphTrainer(params, progress_bar=True).train(dataset, method="T-MPHN")
print(run.best)

# repeated evaluation: tune once, save the parameters, repeat over 30 seeds
evaluation = Evaluation(params, task="graph", monitor="roc_auc", params_dir="./params/adhd")
evaluation.tune_and_evaluate(dataset, "T-Spectral", grid, seeds=range(42, 72), maxiter=30)
evaluation.compare(dataset, ["T-MPHN", "T-Spatial", "GATConv", "GIN"], seeds=range(42, 72))

# tables, Excel workbook and figures
affichage = Affichage(outdir="./exports", show=False)
affichage.report(evaluation.runs, name="adhd_s100", searches=evaluation.searches)
print(affichage.summary_text(evaluation.runs))
```

Methods are selected by name:

- Tensor models: `"T-MPHN"`, `"T-Spectral"`, `"T-Spatial"`
- GIN baseline: `"GIN"`
- NeuroGraph baselines (`ResidualGNNs`): `"GCNConv"`, `"GATConv"`, `"SAGEConv"`, and the other layers listed in `GRAPH_METHODS`

## Evaluation Protocol

- **Splits:** stratified train/validation/test, one split per seed. `cross_validate` gives stratified k-fold instead.
- **Model selection:** epochs and hyperparameters are chosen on the validation set only. Test scores are reported at the epoch with the best validation score (`monitor`, default `roc_auc`).
- **Metrics:** accuracy, balanced accuracy, AUC-ROC, F1, precision, sensitivity, specificity, MCC, PR-AUC and NPV.
  - Binary tasks use class 1 as the positive class.
  - For more than two classes, F1 and precision are macro averages, AUC is one-vs-rest, and the binary-only metrics are left out.
- **Reproducibility:** `params.seed` controls the split, the weight initialization and the hyperedge sampling. The parameters of each full evaluation are saved as JSON in `params/`, so experiments can be rerun without a new grid search.
- **Reporting:**
  - Mean ± std over runs for every dataset, method and metric.
  - Methods × datasets tables.
  - Paired Wilcoxon tests with Holm correction.
  - Learning curves, ROC curves, confusion matrices, runtime and search plots.

## Repository Structure

```text
.
├── run_adhd.py               # ADHD (Schaefer100, AAL116, PP264): tune once, then 30 seeds
├── run_hcp.py                # NeuroGraph HCP datasets: tune once, then 30 seeds
├── run_hcp_single.py         # One run per method on HCP for quick tests
├── run_node.py               # Node classification on co-authorship / co-citation hypergraphs
├── smoke_test.py             # Checks that every method trains on synthetic data
├── src/
│   ├── graph_trainer.py      # GraphTrainer: graph classification for every method
│   ├── node_trainer.py       # NodeTrainer: node classification (T-models + MLP baseline)
│   ├── base_trainer.py       # Shared training loop, seeding, splits, parameter I/O
│   ├── scoring.py            # Scoring: metrics, per-epoch and per-run results
│   ├── evaluation.py         # Evaluation: grid/exhaustive search, repeated runs, CV, saved params
│   ├── affichage.py          # Affichage: summary tables, Excel export, figures
│   ├── config.py             # Params, TMPHNConfig, GridSearchParams
│   ├── thgnn/                # Tensorized HyperGNNs
│   │   ├── structure.py      #   hyperedge construction and term compilation
│   │   ├── data.py           #   hypergraph containers and loaders
│   │   ├── layers.py         #   T-MPHN layer
│   │   ├── model.py          #   T-MPHN
│   │   ├── tspectral.py      #   T-Spectral
│   │   └── tspatial.py       #   T-Spatial
│   ├── gnn.py                # GIN baseline
│   ├── residual_gnn.py       # NeuroGraph ResidualGNNs baselines
│   ├── models/               # Original T-HyperGNN implementations (reference)
│   └── utils/                # Datasets (NeuroGraph, ADHD), preprocessing, edge construction
├── data/
│   ├── adhd/                 # ADHD ROI time series per atlas and targets.csv
│   ├── hcp_data/             # NeuroGraph HCP datasets
│   └── academic_net/         # Hypergraphs for node classification
├── params/                   # Saved parameter sets per dataset and method
└── exports/                  # Excel workbooks and figures
```

## Research Objective

The goal of this work is to study tensorized hypergraph neural networks for brain network analysis.

Brain networks can naturally be represented using graph or hypergraph structures, where nodes represent brain regions and edges or hyperedges represent relationships between them. Tensorized HyperGNNs model joint (higher-order) interactions among groups of regions, which may improve graph-level prediction tasks such as diagnosis or phenotype prediction.

The current research questions include:

- How do T-MPHN, T-Spectral and T-Spatial compare with standard GNN baselines on ADHD and NeuroGraph datasets?
- How do hyperedge construction (neighborhood, kNN, top-k by connectivity strength) and the hypergraph order M affect performance?
- How does tensorized message passing capture higher-order brain network relationships?
- How can the computational cost be reduced further for high-resolution parcellations?

## Current Limitations

- **Memory for T-Spectral and T-Spatial.** Both need O(N³) memory per subject. They are practical on the ADHD atlases (100 to 264 ROIs), but not on 1000-ROI HCP parcellations. Use T-MPHN there.
- **Long HCP runs.** Repeated evaluations on the large HCP datasets (especially HCPTask) take a long time.
- **Regression tasks are not supported.** HCPFI and HCPWM have continuous targets.
- **Some NeuroGraph baselines may not run.** `MLP`, `GINConv` and `APPNP` depend on how `ResidualGNNs` builds its layers.
- **Environment-specific settings.** Run scripts contain experiment-specific paths and parameters that may need to be changed for another environment.