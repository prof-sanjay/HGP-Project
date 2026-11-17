"""
Graph assembly and storage.

Convention everywhere: G[i, j] is the weight of the directed edge source j -> target i; zero diagonal.
"""

from pathlib import Path

import numpy as np
import torch


def standardised_effect(psi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """a[i, j] = |Psi_h[i, j]| sigma_j / sigma_i, zero diagonal."""
    a = np.abs(psi) * (sigma[None, :] / sigma[:, None])
    np.fill_diagonal(a, 0.0)
    return a


def var_top_k(psi: np.ndarray, k: int) -> np.ndarray:
    """The k sources with the largest |Psi_h| per target, weighted by |Psi_h|; sets the weight scale of G_h."""
    a = np.abs(psi).astype(np.float64)
    np.fill_diagonal(a, -np.inf)
    k = min(k, len(a) - 1)
    top = np.argsort(-a, axis=1, kind="stable")[:, :k]
    graph = np.zeros_like(a)
    rows = np.repeat(np.arange(len(a)), k)
    graph[rows, top.ravel()] = a[rows, top.ravel()]
    return np.maximum(graph, 0.0)


def assemble(selected: np.ndarray, effect: np.ndarray, reference: np.ndarray):
    """
    G_h: every knockoff-selected edge weighted by its standardised impulse response, then one global factor so that
    the mean edge weight equals that of the top-k impulse-response graph of the same horizon (`reference`).
    """
    selected = selected.copy()
    np.fill_diagonal(selected, False)
    w = np.where(selected, effect, 0.0).astype(np.float64)
    factor = float(reference[reference > 0].mean() / w[w > 0].mean()) if (w > 0).any() else 1.0
    w *= factor
    degree = (w > 0).sum(axis=1)
    info = {"edges": int((w > 0).sum()), "selected": int(selected.sum()), "scale_factor": factor,
            "sources_per_target": float(degree.mean()), "targets_without_sources": int((degree == 0).sum())}
    return w.astype(np.float32), info


def save_graph(path: Path, graph: np.ndarray, sensors: np.ndarray, horizon: int, train_end: int, **meta) -> None:
    np.savez_compressed(path, adjacency=graph.astype(np.float32), sensors=np.asarray(sensors, np.int64), horizon=horizon,
                        train_end=train_end, orientation="target_source", **meta)


def load_graph(path: Path, horizon: int) -> np.ndarray:
    with np.load(path) as z:
        if int(z["horizon"]) != horizon:
            raise ValueError(f"{path}: graph is for horizon {int(z['horizon'])}, not {horizon}")
        g = np.asarray(z["adjacency"], dtype=np.float32)
    np.fill_diagonal(g, 0.0)
    return g


def load_graphs(graph_dir: Path, horizons) -> dict:
    """The proposed assignment: G_h in the branch of horizon h."""
    return {h: load_graph(Path(graph_dir) / f"G_h{h}.npz", h) for h in horizons}


def row_normalise(graph: np.ndarray) -> np.ndarray:
    """Normalise over incoming edges: every target with at least one source gets weights summing to one."""
    degree = graph.sum(axis=1, keepdims=True)
    return np.where(degree > 0, graph / np.maximum(degree, 1e-8), 0.0).astype(np.float32)


def to_sparse(graph: np.ndarray, device) -> torch.Tensor:
    rows, cols = np.nonzero(graph)
    idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.int64)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(graph[rows, cols]), graph.shape, device=device,
                                   check_invariants=True).coalesce()
