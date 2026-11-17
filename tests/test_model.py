import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hgp.config import ModelConfig
from hgp.data import window_index
from hgp.graphs import row_normalise, to_sparse
from hgp.metrics import masked_mae
from hgp.model import HICP, SemanticGraph, count_parameters

H, N, B, T = (3, 6, 12), 7, 2, 12


def graphs(seed):
    rng = np.random.default_rng(seed)
    return {h: to_sparse(row_normalise((rng.random((N, N)) < 0.5) * rng.random((N, N))), "cpu") for h in H}


def road():
    r = np.zeros((N, N), dtype=np.float32)
    for i in range(N - 1):
        r[i, i + 1] = r[i + 1, i] = 1.0
    return to_sparse(row_normalise(r), "cpu")


def model():
    torch.manual_seed(0)
    return HICP(5, H, ModelConfig(hidden=16, horizon_embed=4, dropout=0.0, semantic_dim=4, semantic_top_k=3), num_nodes=N)


def test_forward_shapes_and_gate_readout():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    pred, gates = model()(flow, tf, graphs(0), road())
    assert pred.shape == (B, 3, N)
    for g in gates.values():
        assert set(g) == {"causal", "physical", "semantic"} and 0 < sum(float(v) for v in g.values()) < 1


def test_gate_is_a_convex_mixture_of_self_and_three_branches():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m = model()
    with torch.no_grad():
        Z = m.encoder(flow, tf)
        S = [m.propagation[0](Z, graphs(0)[3]), m.physical_propagation[0](Z, road()), m.semantic_propagation[0](Z, m.semantic_graph())]
        alpha = m.gate(Z, S, 0)
        expected = m.heads[0](alpha[:, :, 0] * Z + sum(alpha[:, :, k + 1] * S[k] for k in range(3))).squeeze(-1)
        pred, _ = m(flow, tf, graphs(0), road())
    assert alpha.shape == (B, N, 4, 16) and torch.allclose(alpha.sum(2), torch.ones(B, N, 16), atol=1e-6)
    assert torch.allclose(pred[:, 0], expected, atol=1e-6)


def test_each_horizon_uses_its_own_graph():
    flow, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m = model()
    with torch.no_grad():
        base, _ = m(flow, tf, graphs(0), road())
        g = graphs(0)
        g[12] = graphs(1)[12]                                    # change only the 60-minute graph
        moved, _ = m(flow, tf, g, road())
    assert torch.allclose(base[:, :2], moved[:, :2]) and not torch.allclose(base[:, 2], moved[:, 2])


def test_parameter_count_does_not_depend_on_the_graphs():
    assert count_parameters(model()) == count_parameters(model())
    m = model()
    assert {n.split(".")[0] for n, _ in m.named_parameters()} == {
        "encoder", "propagation", "physical_propagation", "semantic_graph", "semantic_propagation", "gate", "heads"}


def test_semantic_graph_is_row_stochastic_top_k_without_self_loops():
    torch.manual_seed(0)
    A = SemanticGraph(N, 4, top_k=3)()
    assert A.shape == (N, N) and torch.allclose(A.sum(1), torch.ones(N), atol=1e-6)
    assert ((A > 0).sum(1) == 3).all() and float(A.diagonal().abs().max().detach()) == 0
    A.sum().backward()                                           # the embeddings receive gradient through the kept entries


def test_window_index_matches_brute_force():
    rng = np.random.default_rng(0)
    rows, seq_len = 400, 12
    input_ok = rng.random(rows) > 0.05
    target_ok = rng.random(rows) > 0.1
    starts, mask = window_index(input_ok, target_ok, seq_len, H)
    for s, m in zip(starts, mask):
        assert input_ok[s:s + seq_len].all()
        for k, h in enumerate(H):
            row = s + seq_len + h - 1
            assert m[k] == (row < rows and target_ok[row])
    expected = [s for s in range(rows - seq_len) if input_ok[s:s + seq_len].all()
                and any(s + seq_len + h - 1 < rows and target_ok[s + seq_len + h - 1] for h in H)]
    assert starts.tolist() == expected


def test_masked_mae():
    pred, target = torch.zeros(2, 3, 4), torch.ones(2, 3, 4)
    target[:, 2] = float("nan")
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.isclose(masked_mae(pred, target, mask), torch.tensor(1.0))
