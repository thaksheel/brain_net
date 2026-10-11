"""T-MPHN: tensor message-passing hypergraph neural network (cached, batched, GPU-friendly)."""
from .structure import (
    CompiledTerms,
    add_self_loops,
    adjacency_value,
    compile_terms,
    hyperedges_from_adjacency,
    hyperedges_from_incidence,
    induced_hyperedges,
    knn_hyperedges,
)
from .data import HyperBatch, HyperGraph, HyperGraphLoader, Terms, build_hypergraph, collate
from .layers import TMPHNLayer, global_pool, normalize, tensor_message
from .model import TMPHN
from .train import (
    evaluate_graphs,
    evaluate_nodes,
    random_split,
    train_graph_classification,
    train_node_classification,
)
