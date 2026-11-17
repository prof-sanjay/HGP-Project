"""
Candidate pool: the sensors within two undirected road hops.

Two sensors are candidates for each other when they share a local unit U_k = {k} + road neighbours of k (both
directions). The pool is both the support of the VAR and the set from which the knockoff filter selects sources.
"""

import numpy as np


def candidate_pool(road: np.ndarray) -> np.ndarray:
    """Boolean [N, N]: pool[i, j] is True when i and j are within two undirected road hops; diagonal False."""
    N = road.shape[0]
    unit = ((road != 0) | (road != 0).T | np.eye(N, dtype=bool)).astype(np.int32)
    pool = (unit @ unit) > 0
    np.fill_diagonal(pool, False)
    return pool
