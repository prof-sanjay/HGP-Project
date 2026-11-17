"""
Part A: one causal graph per forecast horizon, from the training rows only.

    python scripts/build_graphs.py --data data/pems_bay.npz --out graphs

Steps: two-hop ridge VAR -> impulse responses Psi_h -> group-knockoff selection of each target's sources at FDR
level q -> edges weighted by the standardised impulse response. For every horizon h this writes
    graphs/G_h{h}.npz          the graph G_h (adjacency[i, j] = weight of source j for target i)
    graphs/G_h{h}.json         settings, edge counts and build time
    graphs/knockoff_h{h}.npz   knockoff statistics of every target and the selected edges
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The lasso path stops early on a few nearly collinear targets; the entry penalties it records are still valid.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from hgp.config import DataConfig, GraphConfig
from hgp.graphs import (assemble, candidate_pool, detrend, fit_var, impulse_responses, knockoff_statistics, save_graph,
                        select_sources, standardised_effect, var_top_k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--out", default=GraphConfig.out_dir)
    args = ap.parse_args()
    dc, gc = DataConfig(path=args.data), GraphConfig(out_dir=args.out)
    out = Path(gc.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(dc.path) as z:
        flow, road, sensors = z["flow"].astype(np.float64), z["road"], z["sensors"]
    train_end = int(len(flow) * dc.train_rate)
    train = flow[:train_end]
    N = flow.shape[1]
    t0 = time.time()

    pool = candidate_pool(road)
    coef = fit_var(train, pool, gc.var_lag, gc.var_ridge)
    psi = impulse_responses(coef, dc.horizons)
    sigma = np.nanstd(train, axis=0)
    sigma[sigma == 0] = 1.0
    detrended = detrend(flow, train_end)
    print(f"VAR fitted ({time.time() - t0:.0f}s); two-hop pool {pool.sum(1).mean():.1f} candidates per target", flush=True)

    for h in dc.horizons:
        t1 = time.time()
        W_grp, W_ind = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
        groups, gammas = {}, np.zeros(N, np.float32)
        for k, i in enumerate(range(N)):
            cand = np.flatnonzero(pool[i])
            if len(cand) == 0:
                continue
            res = knockoff_statistics(detrended, i, cand, h, train_end, gc, np.random.default_rng(1000 + int(i)))
            if res is None:
                continue
            W_grp[i, cand], W_ind[i, cand], groups[i], gammas[i] = res
            if (k + 1) % 100 == 0:
                print(f"  h={h}: {k + 1}/{N} targets ({time.time() - t1:.0f}s)", flush=True)

        ind_scale = float(np.abs(W_ind).max())
        selected = np.zeros((N, N), dtype=bool)
        for i, grp in groups.items():
            cand = np.flatnonzero(pool[i])
            keep = select_sources(W_grp[i, cand].astype(np.float64), W_ind[i, cand].astype(np.float64), grp, gc.fdr,
                                  gc.max_sources, ind_scale)
            selected[i, cand[keep]] = True

        graph, info = assemble(selected, standardised_effect(psi[h], sigma), var_top_k(psi[h], gc.max_sources))
        info.update(horizon=h, targets_done=len(groups), seconds=time.time() - t1, graph_config=asdict(gc))
        save_graph(out / f"G_h{h}.npz", graph, sensors, h, train_end, fdr=gc.fdr)
        np.savez_compressed(out / f"knockoff_h{h}.npz", W_group=W_grp, W_individual=W_ind, gamma=gammas, selected=selected,
                            sensors=sensors)
        (out / f"G_h{h}.json").write_text(json.dumps(info, indent=2))
        print(f"h={h:2d}: {info['edges']} edges, {info['sources_per_target']:.1f} sources/target, "
              f"{info['targets_without_sources']} targets without sources ({info['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
