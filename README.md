# Tensorized Hypergraph GNN for Brain Network Classification

This repository contains a modified implementation of a Tensorized Hypergraph Graph Neural Network (Tensorized HyperGNN) for graph classification on brain network datasets.

The original model was designed for node classification. This version modifies the model and training pipeline to support graph classification, with a focus on NeuroGraph datasets used in brain network analysis.

## Project Status

This project is an active research implementation. The current focus is evaluating the modified Tensorized HyperGNN on NeuroGraph datasets, particularly brain network datasets.

The NeuroGraph dataset pipeline has been integrated successfully. A small test dataset for graph classification experiment can be started using [`run_gcls.py`](run_gcls.py). However, the NeuroGraph datasets are large, and the current model is computationally expensive. Training can therefore take a substantial amount of time for each epoch. You can try [`run_ng.py`](run_ng.py) and observe the high runtime and memory requirement. 

Performance optimization and large-scale NeuroGraph experiments are ongoing and the current focus.

## Tensorized HyperGNN Models

The initial Tensorized HyperGNN implementation includes three model variants:

- **T-Spatial**
- **T-Spectral**
- **T-Message Passing (T-MPHN)**

For the graph classification implementation in this repository, only **T-MPHN** is currently working correctly. The T-Spatial and T-Spectral implementations were originally developed for node classification and have not yet been fully adapted or validated for graph classification.

## Repository Structure

```text
.
├── run_ng.py                 # Entry point for NeuroGraph graph classification
├── run_time_test.py          # Runtime experiments on TU datasets
├── src/
│   ├── train.py              # Training and evaluation pipeline
│   ├── config.py             # Model and experiment parameters
│   ├── models/               # Tensorized HyperGNN models
│   ├── utils/                # Dataset and tensor utilities
│   └── neighbors.py          # Neighborhood construction
├── data/
│   ├── TUDataset/            # TU benchmark datasets
│   └── edge/                # Brain network edge data
├── exports/                  # Experiment outputs
└── temp/                     # Experimental scripts and notes