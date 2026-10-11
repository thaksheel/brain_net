# T-MPHN, reimplemented

A cached, batched, GPU-friendly reimplementation of the tensor message-passing hypergraph neural network (T-MPHN) from *T-HyperGNNs: Hypergraph Neural Networks via Tensor Representations* (Wang et al.).

The message math (adjacency values, edge spanning, permutation weights, Hadamard cross-node interactions, degree normalization) is the same as the original code, and `tests/test_structure.py` checks it against a line-by-line port of the original aggregator. The engineering is new: no recursion, no Python loops in the forward pass, and each layer runs once for all nodes.

## Quick start

```bash
pip install torch numpy            # scipy only needed for sparse incidence matrices
python tests/test_structure.py                                  # NumPy-only checks
python tests/test_torch.py path/to/original/tmessagepassing.py  # PyTorch checks (+ comparison with the original)
python -m tmphn.train                                           # synthetic demo, node and graph tasks
```

## How it works

```
hyperedges ──compile_terms(M)──► Terms            once per hypergraph and order M  (NumPy, structure.py)
                                    │
x ──Linear──normalize──► h0          ▼
h0 ──TMPHNLayer(Mlst[0])──► h1      per layer, for ALL nodes at once  (torch, layers.py):
...                                   m = scatter_add( coef * h[idx0] * h[idx1] * ... )
h(L-1) ──TMPHNLayer(Mlst[L-1])──► hL   h' = normalize(ReLU(Linear(dropout([h, m]))))
hL ──(pool, graph task)──Linear──► log_softmax
```

**Terms.** A term is one summand of the neighborhood message, `message[target] += coef * h[i1] * ... * h[i(M-1)]`. Compiling the whole neighborhood table into three flat arrays (`target`, `idx`, `coef`) up front turns every layer into a gather, an elementwise product, and a scatter-add. All three map well onto the GPU.

## Where things moved

| Original | New |
|---|---|
| `Neighbors.py` (`NeighborFinder`) | `structure.py`: hyperedge builders, `choose_members` (down-sampling), `compile_terms` |
| `TMessagePassing.aggregate_with_M / aggregate_with_c / adj_coef` | `structure.expansion_patterns`, `adjacency_value` (computed once), and `layers.tensor_message` (per layer) |
| `Encoder` | `layers.TMPHNLayer` |
| `TMPHN.py` | `model.TMPHN` |
| T-MPHN branch of `prepare.py`, `Encoder.forward_with_features`, `config.HypergraphBatchLoader` | `data.build_hypergraph`, `collate`, `HyperGraphLoader` |
| `prepare.evaluate` (T-MPHN part) | `train.evaluate_nodes`, `evaluate_graphs` |

## Issues resolved

| Issue in the original | Resolution |
|---|---|
| Layers chained through `features` callables, giving hidden recursion and neighborhood explosion (about `f^(L-1)` lower-layer evaluations per node) | Layers are an `nn.ModuleList` run in a `for` loop. Each layer runs once for all nodes. |
| Self embedding computed twice per layer (once only for `zeros_like`) | Computed once. |
| Python loops over nodes, edges, and combinations in every forward pass | Combinatorics done once at compile time. The forward pass is vectorized. |
| Dropout re-sampled in every recursive call, then multiplied inside Hadamard products | Dropout applied once, on the linear layer's input. Messages use the un-dropped `h`, so dropout masks are never multiplied together. |
| No input MLP and no normalization, so products of raw features vanish or explode | Input projection (`x0 = MLP(x)`, Algorithm 2) plus per-layer normalization. |
| Edges larger than a layer's `M` silently skipped (but still counted in the degree) when `Mlst[l] < args.M` | Every layer down-samples oversized edges to its own `M`, as in paper Sec. IV-D. |
| `NeighborFinder`: `random.seed(42)` reset on every call, global RNG shared across dask threads, full edge scan per node | Vectorized NumPy with a seeded `Generator`. No dask. |
| Random down-sampling only | `strategy="random" \| "topk" \| "weighted"`, using a `scores` matrix (e.g. abs(FC)). |
| Isolated nodes returned their own features as the "message" | Message is 0. Self information still enters through `combine`. |
| Lambda late-binding bug when building layers in a loop | No closures. |
| Earlier layers registered under several names, so `state_dict()` was bloated | Each parameter appears once. |
| Scattered `.to(self.device)`, with `X` left on CPU | Call `.to(device)` on the model and the data. Every op follows its inputs' device. |
| Per-graph features passed through `forward_with_features` | `x` and `terms` are explicit arguments. Graphs are batched as a disjoint union. |

## Intentional behavior changes

- **`norm="rms"` by default.** Unit root-mean-square per node keeps entries O(1), so products of `M-1` entries stay well scaled. `norm="l2"` is the paper's Eq. 26, but there entries are about `1/sqrt(D)`, so messages shrink like `D^-(M-1)/2`. `norm="none"` is the original behavior. If training is unstable with large `M`, try `"l2"`.
- **`project_input=True` by default.** `False` feeds raw `x` to layer 0, like the original. Combined with `norm="none"`, this gives a close ablation of the original architecture.
- **Shared sampling across layers.** Layers with the same `M` share one compiled (and sampled) structure.

## Usage

### Node classification on one hypergraph (paper setting)

```python
from tmphn import TMPHN, build_hypergraph, hyperedges_from_incidence, random_split, train_node_classification

edges = hyperedges_from_incidence(H)                 # H: (N, E) incidence matrix
graph = build_hypergraph(X, edges, Mlst=[5, 3], y=Y, self_loops=False, seed=0)
model = TMPHN(in_dim=X.shape[1], hid_dim=256, num_classes=int(Y.max()) + 1, Mlst=[5, 3])
print(train_node_classification(model, graph, random_split(len(Y)), epochs=200))
```

`TMPHN.from_params(params)` builds the model from your `config.Params`.

### Graph classification: one hypergraph per subject (brain data)

```python
import numpy as np
from tmphn import TMPHN, build_hypergraph, knn_hyperedges, train_graph_classification

A_np, X_np, y_np = from_pygeo_to_matrices(dataset)   # your config.py helper
graphs = []
for i in range(len(y_np)):
    fc = X_np[i]                                     # ROI x ROI connectivity, also used as node features
    edges = knn_hyperedges(np.abs(fc), k=4)          # each ROI + its 4 most strongly connected ROIs
    graphs.append(build_hypergraph(fc, edges, Mlst=[5, 3], y=y_np[i],
                                   strategy="topk", scores=np.abs(fc)))   # layer 2 keeps the strongest members

model = TMPHN(in_dim=X_np.shape[2], hid_dim=128, num_classes=2, Mlst=[5, 3], task="graph", pool="mean")
print(train_graph_classification(model, graphs[:train_n], graphs[train_n:], batch_size=16))
```

Other builders: `hyperedges_from_adjacency(A)` (each node plus its graph neighbors), `hyperedges_from_incidence(H)`, `add_self_loops`.

Compute `scores` from training data only if they are group-level, to avoid leakage. Per-subject FC is fine because it is that subject's own input.

### Inductive training

Compile the training structure from the visible nodes only, and evaluate with the full structure:

```python
train_edges = induced_hyperedges(edges, nodes=train_and_val_nodes, num_nodes=N)
g_train = build_hypergraph(X, train_edges, Mlst, y=Y)
g_full = build_hypergraph(X, edges, Mlst, y=Y)
```

### Resampling neighbors

Call `build_hypergraph(..., seed=epoch)` again every few epochs. Compilation is vectorized: about 0.1 s for a ModelNet40-sized hypergraph (12k nodes, 25k edges).

## Experiment scripts

These go in your project root, next to `src/` and `tmphn/`:

| Script | Task | Data |
|---|---|---|
| `run_node_classification.py` | Node classification | Academic hypergraphs (`hypergraph.pickle`, `feature.pickle`, `labels.pickle`) |
| `run_adhd.py` | Graph classification | ADHD atlases through `src.Preprocess` / `src.GraphDataset` |
| `run_neurograph.py` | Graph classification | NeuroGraph HCP datasets through `src.utils.datasets.NeuroGraphDataset` |

The two graph scripts share `graph_cls_utils.py`. It handles the conversion of graphs to hypergraphs (`hyperedges = "edges" | "neighborhood" | "knn"`), the stratified split, per-epoch metrics and timing, and the Excel and PNG export. Converting a 1000-ROI subject with neighborhood hyperedges and top-k down-sampling takes about 0.1 s and is done once per run.

## Large graphs and GPU notes

- **Number of terms per layer:** `T = sum over edges of |e| * C(M-1, |e|-1)` for `|e| <= M`, and `|e|` for larger edges. Check it with `graph.terms[l].num_terms`.
- **Activation memory per layer:** without checkpointing, roughly `(2M-3) * T * D` floats.
- **Reducing memory:** if that is too large, use `TMPHN(..., chunk_size=65536, checkpoint=True)`. Chunk products are then recomputed during backward instead of stored, so peak memory drops to about `N * D` per layer, at the cost of about one extra forward pass.
- **Where to compile:** compile on CPU once, then move each graph (or batch) to the GPU with `.to(device)`. Small datasets can be moved to the GPU up front.
- **Limitation:** training is full-batch per graph. A single hypergraph whose terms do not fit even with checkpointing would need subgraph mini-batching, which is not included.
- **Determinism:** `index_add` on CUDA uses atomic adds, so results can differ in the last bits between runs.

## Verification status

- `tests/test_structure.py` (NumPy) passes. It covers exact agreement with the original aggregator for M = 2 to 5 including singleton edges, the paper's Fig. 2 example, the adjacency values of Fig. 5, down-sampling (seeded, top-k), and the builders.
- `tests/test_torch.py` covers the message against the reference and the original, chunking and checkpointing (values and gradients), batched vs. single-graph outputs, a decreasing training loss, `state_dict` aliasing, and CPU vs. CUDA. **These torch tests were written without access to PyTorch, so run them before relying on the code.**

## Pitfalls in the current `config.py`

- **`type=bool` in argparse:** `--self_loop False` parses as `True`, because `bool("False")` is `True`. Use `action="store_true"` or a str-to-bool converter.
- **`--Mlst` with `type=list`:** `--Mlst 53` becomes `['5', '3']`, a list of characters. Use `nargs="+", type=int`, which gives `--Mlst 5 3`.
- **`combine` docstring:** it says it merges "multiple hyperedge messages". It actually merges a node's own vector with its aggregated message.
