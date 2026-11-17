import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("tables")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import DataConfig
from hgp.data import load

ROOT = Path(__file__).resolve().parents[1]


def make_pems_bay(directory: Path, n=10, steps=3000):
    """Files in the layout of the DCRNN release: speed table in HDF5 (0 = missing), pickled [ids, id_to_index, adj]."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2017-01-01", periods=steps, freq="5min")
    speed = 60 + 5 * rng.standard_normal((steps, n))
    speed[rng.random((steps, n)) < 0.01] = 0.0
    pd.DataFrame(speed, index=index, columns=[str(i) for i in range(n)]).to_hdf(directory / "pems-bay.h5", key="speed")
    adj = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = 0.8
    with open(directory / "adj_mx_bay.pkl", "wb") as f:
        pickle.dump([[str(i) for i in range(n)], {str(i): i for i in range(n)}, adj], f)


def test_pems_bay_preparation_and_split(tmp_path):
    make_pems_bay(tmp_path)
    out = tmp_path / "bay.npz"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--traffic", str(tmp_path / "pems-bay.h5"),
                           "--adjacency", str(tmp_path / "adj_mx_bay.pkl"), "--out", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "two-hop candidate pool" in proc.stdout
    with np.load(out) as z:
        assert z["flow"].shape == (3000, 10)
        assert np.isnan(z["flow"]).any() and not (z["flow"] == 0).any()          # zeros become missing
        assert np.isfinite(z["inputs"]).mean() > np.isfinite(z["flow"]).mean()   # short gaps filled in the inputs only
        assert z["timestamps"][1] - z["timestamps"][0] == 300_000_000_000
    data = load(DataConfig(path=str(out)))
    assert data.train_end == 2100 and data.val_end == 2400 and data.num_nodes == 10   # 70 / 10 / 20
