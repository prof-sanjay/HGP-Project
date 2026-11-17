"""
Two-hop restricted ridge VAR and its impulse responses.

    x_t = sum_{k=1}^{p} A_k x_{t-k} + e_t,   (A_k)[i, j] = 0 unless j = i or j is in the candidate pool of i

The support of every equation is the target itself and the sensors within two road hops (hgp/graphs/pool.py), so every
edge the knockoff filter can select carries an impulse-response weight. Each target equation is fitted separately on
the rows where the target and all its lagged predictors are observed; missing values are never filled.
"""

import numpy as np


def fit_var(flow: np.ndarray, support: np.ndarray, lag: int = 3, ridge: float = 1e6) -> np.ndarray:
    """flow [T, N] training rows with NaN; support [N, N] (the candidate pool).
    Returns coefficients [lag, N, N] with A_k[i, j] = effect of sensor j at lag k on sensor i."""
    T, N = flow.shape
    coef = np.zeros((lag, N, N), dtype=np.float64)
    for i in range(N):
        sources = np.union1d(np.flatnonzero(support[i] != 0), [i])
        X = np.concatenate([flow[lag - k:T - k][:, sources] for k in range(1, lag + 1)], axis=1).astype(np.float64)
        y = flow[lag:, i].astype(np.float64)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if not ok.any():
            continue
        X, y = X[ok], y[ok]
        beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
        for k in range(lag):
            coef[k, i, sources] = beta[k * len(sources):(k + 1) * len(sources)]
    return coef


def impulse_responses(coef: np.ndarray, horizons) -> dict:
    """Psi_0 = I, Psi_s = sum_{k<=min(s,p)} A_k Psi_{s-k}; returns {h: Psi_h}."""
    p, N = coef.shape[0], coef.shape[1]
    psi = [np.eye(N)]
    for s in range(1, max(horizons) + 1):
        psi.append(sum(coef[k - 1] @ psi[s - k] for k in range(1, min(s, p) + 1)))
    return {int(h): psi[h] for h in horizons}
