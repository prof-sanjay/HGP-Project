"""End-to-end check of the scripts on a small synthetic dataset: prepared file -> graphs -> training -> summary."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_dataset(path: Path, n=12, days=40, seed=0):
    """A chain of sensors in which every sensor follows its upstream neighbour three steps later."""
    rng = np.random.default_rng(seed)
    T = days * 288
    road = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        road[i, i + 1] = road[i + 1, i] = 1.0
    daily = 50 + 20 * np.sin(2 * np.pi * np.arange(T) / 288)[:, None]
    flow = daily + rng.standard_normal((T, n)) * 5
    for t in range(3, T):
        flow[t, 1:] += 0.3 * (flow[t - 3, :-1] - daily[t - 3, 0])
    flow = flow.astype(np.float32)
    flow[rng.random((T, n)) < 0.01] = np.nan
    inputs = np.where(np.isfinite(flow), flow, np.nanmean(flow, 0)).astype(np.float32)
    timestamps = 1_609_459_200_000_000_000 + np.arange(T, dtype=np.int64) * 300_000_000_000
    np.savez_compressed(path, flow=flow, inputs=inputs, timestamps=timestamps, sensors=np.arange(n), road=road)


def run(*args):
    proc = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


def test_pipeline_end_to_end(tmp_path):
    data, graphs, results = tmp_path / "toy.npz", tmp_path / "graphs", tmp_path / "results"
    make_dataset(data)
    run("scripts/build_graphs.py", "--data", str(data), "--out", str(graphs))
    for h in (3, 6, 12):
        g = np.load(graphs / f"G_h{h}.npz")
        info = json.loads((graphs / f"G_h{h}.json").read_text())
        assert int(g["horizon"]) == h and info["edges"] == int((g["adjacency"] > 0).sum())
        assert info["graph_config"]["fdr"] == 0.3
    # the toy sensors follow their upstream neighbour 3 steps later, so sources are found at h = 3
    assert json.loads((graphs / "G_h3.json").read_text())["edges"] > 0

    for seed in (1, 2):
        run("scripts/train.py", "--seed", str(seed), "--data", str(data), "--graphs", str(graphs), "--results", str(results),
            "--epochs", "2", "--max-windows", "512", "--device", "cpu")
    r = json.loads((results / "proposed__seed1" / "results.json").read_text())
    assert set(r["test"]) == {"3", "6", "12", "mean"} and r["best_epoch"] >= 1
    assert r["branches"] == ["causal", "physical", "semantic"]
    for h in ("3", "6", "12"):
        assert 0 < sum(r["test_gate"][h].values()) < 1
        z = np.load(results / "proposed__seed1" / f"test_predictions_h{h}.npz")
        assert not np.isnan(z["targets"]).any() and z["targets"].shape == z["predictions"].shape
    history = (results / "proposed__seed1" / "history.csv").read_text().splitlines()[0]
    assert "gate_causal_h3" in history and "gate_semantic_h12" in history

    summary = run("scripts/summarise.py", "--results", str(results))
    assert "2 seed(s)" in summary and "60 min" in summary and "±" in summary
