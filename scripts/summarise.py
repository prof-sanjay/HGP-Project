"""
Test errors of the finished runs: per horizon, the mean and standard deviation over seeds.

    python scripts/summarise.py --results results
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    runs = {int(p.parent.name.rsplit("__seed", 1)[1]): json.loads(p.read_text())
            for p in sorted(Path(args.results).glob("proposed__seed*/results.json"))}
    if not runs:
        raise SystemExit(f"no finished runs under {args.results}")

    horizons = [h for h in next(iter(runs.values()))["test"] if h.isdigit()]
    print(f"proposed model, {len(runs)} seed(s): {sorted(runs)}; test split; MAE and RMSE in mph, MAPE in %")
    print("horizon        " + "  ".join(f"{m.upper():>15}" for m in ("mae", "rmse", "mape")))
    for h in horizons + ["mean"]:
        label = f"{int(h) * 5} min" if h != "mean" else "mean"
        cells = []
        for metric in ("mae", "rmse", "mape"):
            v = np.array([r["test"][h][metric] for r in runs.values()])
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        print(f"{label:<15}" + "  ".join(f"{c:>15}" for c in cells))


if __name__ == "__main__":
    main()
