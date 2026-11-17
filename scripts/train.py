"""
Part B: train the proposed model (HICP with G_3 / G_6 / G_12) and evaluate it on the test split.

    python scripts/train.py --data data/pems_bay.npz --graphs graphs --seed 42

Writes results/proposed__seed<seed>/ with config.json, history.csv, results.json, checkpoints/best.pt and
test_predictions_h{3,6,12}.npz.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import DataConfig, GraphConfig, TrainConfig
from hgp.data import load
from hgp.graphs import load_graphs
from hgp.train import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--graphs", default=GraphConfig.out_dir, help="folder written by scripts/build_graphs.py")
    ap.add_argument("--results", default=TrainConfig.results_dir)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--max-windows", type=int, default=None, help="use only the first K windows per split, for a quick check")
    args = ap.parse_args()

    dc = DataConfig(path=args.data)
    tc = TrainConfig(seed=args.seed, epochs=args.epochs, results_dir=args.results)
    data = load(dc)
    graphs = load_graphs(args.graphs, dc.horizons)
    trainer = Trainer(data, dc, tc, graphs, torch.device(args.device) if args.device else None, args.max_windows)
    trainer.fit()


if __name__ == "__main__":
    main()
