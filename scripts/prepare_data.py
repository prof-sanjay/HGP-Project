"""
Prepare PEMS-BAY for HGP.

    python scripts/prepare_data.py --traffic pems-bay.h5 --adjacency adj_mx_bay.pkl --out data/pems_bay.npz

Input: the DCRNN release of PEMS-BAY (speed in mph, a reading of 0 means missing; the adjacency is the pickled
[sensor_ids, id_to_index, adj_mx]). Writes one .npz with
    flow        observed speed, NaN where missing (targets)
    inputs      the same with gaps of at most 30 minutes linearly interpolated inside each split (model inputs only)
    timestamps  nanoseconds since the epoch, 5-minute grid
    sensors     sensor indices
    road        road adjacency (road[i, j] != 0: a road link between i and j)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import DataConfig
from hgp.graphs.pool import candidate_pool


def interpolate_short_gaps(x: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Linear interpolation of interior gaps of at most max_gap steps; longer gaps stay missing."""
    out = x.copy()
    idx = np.arange(len(x))
    for j in range(x.shape[1]):
        col = out[:, j]
        missing = ~np.isfinite(col)
        if not missing.any() or missing.all():
            continue
        filled = np.interp(idx, idx[~missing], col[~missing])
        edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start <= max_gap and start > 0 and end < len(col):
                col[start:end] = filled[start:end]
    return out


def read_pems_bay(traffic: str, adjacency: str):
    import pandas as pd
    table = pd.read_hdf(traffic)
    flow = table.values.astype(np.float32)
    flow[flow == 0] = np.nan
    timestamps = table.index.values.astype("datetime64[ns]").astype(np.int64)
    with open(adjacency, "rb") as f:
        content = pickle.load(f, encoding="latin1")
    road = np.asarray(content[2] if isinstance(content, (list, tuple)) else content, dtype=np.float32)
    return flow, timestamps, road


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--traffic", required=True, help="pems-bay.h5")
    ap.add_argument("--adjacency", required=True, help="adj_mx_bay.pkl")
    ap.add_argument("--out", default=DataConfig.path)
    ap.add_argument("--subset", type=int, default=None, help="keep only K sensors, for a quick check of the pipeline")
    args = ap.parse_args()

    flow, timestamps, road = read_pems_bay(args.traffic, args.adjacency)
    if road.shape != (flow.shape[1],) * 2:
        raise SystemExit(f"adjacency {road.shape} does not match {flow.shape[1]} sensors")
    dc = DataConfig()
    T = len(flow)
    train_end = int(T * dc.train_rate)
    val_end = train_end + int(T * dc.val_rate)

    sensors = np.arange(flow.shape[1])
    if args.subset:
        sensors = np.sort(np.random.default_rng(0).choice(sensors, min(args.subset, len(sensors)), replace=False))
    flow, road = flow[:, sensors], road[np.ix_(sensors, sensors)]

    inputs = np.empty_like(flow)
    for lo, hi in ((0, train_end), (train_end, val_end), (val_end, T)):
        inputs[lo:hi] = interpolate_short_gaps(flow[lo:hi])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, flow=flow, inputs=inputs, timestamps=timestamps, sensors=np.asarray(sensors, np.int64), road=road)

    links = (road != 0) | (road != 0).T
    np.fill_diagonal(links, False)
    pool = candidate_pool(road)
    print(f"{args.out}: {T} steps x {flow.shape[1]} sensors; missing {np.isnan(flow).mean():.4%} of values, "
          f"{np.isnan(inputs).mean():.4%} after filling")
    print(f"split: train {train_end}, validation {val_end - train_end}, test {T - val_end} steps")
    print(f"road links per sensor {links.sum(1).mean():.1f}; two-hop candidate pool per sensor {pool.sum(1).mean():.1f}")


if __name__ == "__main__":
    main()
