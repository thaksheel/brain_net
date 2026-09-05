import numpy as np
import pandas as pd
import pickle
import json
import torch
import re
import os
from typing import List, Literal, Dict, Annotated
from numpy.typing import NDArray
from dataclasses import dataclass
from sklearn.decomposition import NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csgraph
from scipy.linalg import eigh


@dataclass
class Node:
    """
    Container for subject-level node features derived from fMRI time series.

    This class stores the raw ROI-major time‑series matrix for a single subject
    along with its corresponding node‑feature representation (e.g., PCA‑reduced
    ROI embeddings). Each row corresponds to a brain region (ROI), and each
    column corresponds to a time point or feature dimension.

    Parameters
    ----------
    patient_id : int
        Unique identifier for the subject.
    data : np.ndarray
        Array of shape ``(T, R)`` or ``(R, T)`` representing the fMRI time
        series, where ``T`` is the number of time samples and ``R`` is the number
        of ROIs. In the preprocessing pipeline this is typically stored in
        ROI-major form ``(T, R)``.
    v : np.ndarray
        Node‑feature (usually vector) of shape ``(R, D)``, where ``D`` is the feature
        dimension. Represents the learned or extracted embedding for each ROI.

    Notes
    -----
    The ``v`` attribute is typically produced by PCA, NMF, or another
    dimensionality‑reduction method applied to the ROI‑major time‑series
    matrix. It is used as the node feature matrix ``X`` in downstream
    hypergraph or GNN models.
    """

    patient_id: int
    data: np.ndarray
    v: np.ndarray


@dataclass
class Edge:
    """
    Container for subject-level hyperedge and adjacency information.

    This class stores the padded ROI‑index matrix from the raw HCP hypergraph
    representation, along with the corresponding adjacency matrix and an
    optional edge‑level feature vector. Padding values of ``-1`` indicate
    unused entries in the hyperedge definition.

    Parameters
    ----------
    patient_id : int
        Unique identifier for the subject.
    data : np.ndarray
        Array of shape ``(R, R)`` containing padded ROI indices from the
        hypergraph definition. Entries equal to ``-1`` represent padding.
    A : np.ndarray
        Adjacency matrix of shape ``(R, R)`` derived from ``data``. Contains
        binary (or weighted) connectivity values between ROIs, without self‑loops.
    v : np.ndarray
        Edge‑feature vector (or matrix) of shape ``(R, 1)`` (or ``(R, D)``),
        depending on the representation used.

    Notes
    -----
    The ``data`` matrix originates from the HCP hypergraph format, where each
    row encodes the ROI indices participating in a hyperedge. The adjacency
    matrix ``A`` is typically constructed by connecting all valid (non‑padded)
    ROI pairs within each hyperedge.
    """

    patient_id: int
    data: np.ndarray
    A: np.ndarray
    v: np.ndarray


class Preprocess:
    def __init__(self, n_rois: int, normalize: Literal["minmax", "standard"]):
        self.n_rois = n_rois
        self.normalize = normalize
        self.patient_ids = None

    def get_hypergraph(self, nodes: List[Node], edges: List[Edge], y: np.ndarray):
        """
        Construct patient-level hypergraph inputs from node features, edge
        definitions, and target labels.

        Parameters
        ----------
        nodes : list of Node
            List of ``Node`` objects, one per patient.
        edges : list of Edge
            List of ``Edge`` objects, one per patient.
        y : np.ndarray
            Target labels of shape ``(N,)`` where ``N`` is the number of patients.

        Returns
        -------
        H : np.ndarray
            Hyperedge incidence matrix of shape ``(N, R)``. Each row encodes
            the ROI participation pattern for a patient, typically derived
            from the padded hyperedge matrix by counting or thresholding
            non‑padded ROI occurrences.
        X : np.ndarray
            Node‑feature matrix of shape ``(N, R)``, depending
            on the dimensionality of ``Node.v``. Represents the ROI‑level
            feature embeddings for each patient.
        Y : np.ndarray
            Target vector of shape ``(N,)`` aligned with ``H`` and ``X``.

        Notes
        -----
        - ``R`` refers to the number of ROIs (brain regions).
        - The construction of ``H`` typically involves collapsing the padded
        hyperedge matrix into a binary or frequency‑based ROI participation
        vector.
        - All returned arrays are ordered consistently across subjects, and patients with inconsistent node features are removed.
        """
        node_row_dim = np.array([n.data.shape[0] for n in nodes]).max()
        exclude_patients = [
            int(n.patient_id) for n in nodes if n.data.shape[0] != node_row_dim
        ]
        exclude_i = [i for i, n in enumerate(nodes) if n.data.shape[0] != node_row_dim]
        H = np.array(
            [n.v.squeeze() for n in edges if n.patient_id not in exclude_patients]
        )
        X = np.array(
            [n.v.squeeze() for n in nodes if n.patient_id not in exclude_patients]
        )
        Y = np.array([y for i, y in enumerate(y) if i not in exclude_i])
        return H, X, Y

    def read_data(
        self,
        target_path: str,
        node_folder: str,
        edge_folder: str,
        edge_rep: Literal["specific_k", "spectral", "cosine_sim", "top_frequency"],
        node_rep: Literal["mean", "pca", "linear", "nmf", "specific_k"],
        sparsity: Literal['0.2', "0.5"],
    ):
        df = pd.read_csv(target_path)
        self.patient_ids = df["IID"].to_numpy()
        y = df["Diagnosis"].to_numpy()
        nodes = self._nodes(
            path=node_folder,
            patient_ids=self.patient_ids,
            normalize=self.normalize,
            rep_type=node_rep,
        )
        edges = self._edges(
            path=edge_folder,
            nodes=[n.data for n in nodes],
            sparsity=sparsity,
            patient_ids=self.patient_ids,
            emd_type=edge_rep,
        )
        return y, nodes, edges

    def _padding(self, arr, fill=-1):
        max_len = self.n_rois
        padded = np.array([row + [fill] * (max_len - len(row)) for row in arr])
        return padded

    def _normalize(self, X: np.ndarray, norm_type: Literal["minmax", "standard"]):
        if norm_type == "minmax":
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        elif norm_type == "standard":
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        else:
            raise ValueError(f"unknown norm_type of: {norm_type}")

    def _pca(self, data: np.ndarray, dim_out: int):
        pca = PCA(n_components=dim_out)
        w = pca.fit_transform(data)
        return w

    def _mean(self, data: np.ndarray, dim_out: int):
        v = data.mean(axis=1)
        v = v.reshape(-1, 1)
        return v

    def _nmf(self, data: np.ndarray, dim_out: int):
        nmf = NMF(
            n_components=dim_out,
            init="random",
            solver="mu",
            beta_loss="frobenius",
            max_iter=300,
            random_state=42,
            tol=1e-4,
        )
        w = nmf.fit_transform(data)
        return w

    def _linear(data: np.ndarray, dim_out: int):
        proj = torch.nn.Linear(data.shape[1], dim_out, bias=False)
        v = proj(torch.tensor(data))
        return v.detach().numpy()

    def node_representation(
        self,
        data: np.ndarray,
        rep_type: Literal["mean", "pca", "linear", "nmf", "specific_k"],
        dim_out: int = 1,
    ):
        methods = {
            "specific_k": self._specific_k,
            "mean": self._mean,
            "pca": self._pca,
            "nmf": self._nmf,
            "linear": self._linear,
        }
        return methods[rep_type](data, dim_out)

    def _spectral_embed(self, A: np.ndarray, k: int = 1):
        np.fill_diagonal(A, 0)
        L = csgraph.laplacian(A, normed=True)
        vals, vecs = eigh(L)
        vecs_k = vecs[:, :k]
        return vecs_k.reshape(-1, 1)

    def _specific_k(self, A: np.ndarray, k: int = 1):
        # TODO: improve how this works along with other representation
        return A[k, :].reshape(-1, 1)

    def _cosine_sim(
        self,
        N: np.ndarray,
        dim_out: int = 1,
        k: int = 10,
        outtype: Literal["binary", "weighted"] = "binary",
    ):
        """
        Edge representation using cosine similarities of node features with shape=(timeseries, rois). Output has shape=(rois, k) and k=1 mostly.
        """
        S = cosine_similarity(N.T)
        np.fill_diagonal(S, 0)
        if k is None:
            h = S.mean(1)
        else:
            idx = np.argsort(-S, axis=1)[:, :k]
            h = np.take_along_axis(S, idx, axis=1).mean(axis=1)
        if outtype == "binary":
            h = (h > h.mean()).astype(int)
        elif outtype == "weighted":
            h = h / h.max()
        return h

    def _top_frequency(
        self,
        hyperedges: np.ndarray,
        dim_out: int = 1,
        outtype: Literal["binary", "weighted"] = "binary",
    ):
        _, n_rois = hyperedges.shape
        freq = np.zeros(n_rois, dtype=int)
        for edge in hyperedges:
            for roi in edge:
                if roi > -1:
                    freq[int(roi)] += 1
        if outtype == "binary":
            h = (freq >= freq.mean()).astype(int)
        elif outtype == "weighted":
            h = freq / freq.max()
        return h

    def edge_representation(
        self,
        M: np.ndarray,
        emd_type: Literal["specific_k", "spectral", "cosine_sim", "top_frequency"],
        dim_out: int = 1,
    ):
        """Returns an edge vector for each hyperedge matrix given"""
        methods = {
            "specific_k": self._specific_k,
            "spectral": self._spectral_embed,
            "cosine_sim": self._cosine_sim,
            "top_frequency": self._top_frequency, #TODO: add outtype to control weighted or binary 
        }
        return methods[emd_type](M, dim_out)

    def to_adjacency(self, M: np.ndarray):
        # FIXME: double check self-loop for AAL116
        I = np.zeros((M.shape[0], M.shape[0]))
        for j, row in enumerate(M):
            for i, val in enumerate(row):
                if val > -1:
                    I[val, j] = 1
        return I

    def _edges(
        self,
        path: str,
        patient_ids: List[int],
        nodes: List[np.ndarray],
        sparsity: str,
        emd_type: Literal["specific_k", "spectral", "cosine_sim", "top_frequency"],
        padding: bool = True,
    ) -> List[Edge]:
        filename = f"{path}/edge_{sparsity}.json"
        with open(filename, "r") as f:
            data = json.load(f)
        edges: List[Edge] = []
        for p, patient_id in enumerate(patient_ids):
            data_i = data[str(patient_id)]
            data_i = self._padding(data[str(patient_id)]) if padding else data_i
            A = self.to_adjacency(data_i)
            M = {
                "specific_k": A,
                "spectral": A,
                "cosine_sim": nodes[p],
                "top_frequency": data_i,
            }[emd_type]
            edges.append(
                Edge(
                    patient_id=patient_id,
                    data=data_i,
                    A=A,
                    v=self.edge_representation(M, emd_type=emd_type),
                )
            )
        return edges

    def _nodes(
        self,
        path: str,
        patient_ids: List[int],
        normalize: Literal["minmax", "standard"],
        rep_type: Literal["mean", "pca", "linear", "nmf", "specific_k"] = "nmf",
    ) -> List[Node]:
        nodes: List[Node] = []
        for idx in patient_ids:
            filename = f"{path}/{idx}.pkl"
            with open(filename, "rb") as f:
                data_ = pickle.load(f)
            data = data_.T
            if normalize:
                data = self._normalize(data, norm_type=normalize)
            nodes.append(
                Node(
                    patient_id=idx,
                    data=data,
                    v=self.node_representation(data, rep_type=rep_type, dim_out=1),
                )
            )
        return nodes

    def _quick_rename(self, folder):
        pattern = re.compile(r"sub-(.*?)_task")
        for filename in os.listdir(folder):
            if filename.endswith(".pkl"):
                match = pattern.search(filename)
                if match:
                    patient_id = match.group(1)
                    old_path = os.path.join(folder, filename)
                    new_path = os.path.join(folder, f"{patient_id}.pkl")
                    os.rename(old_path, new_path)
        return True
