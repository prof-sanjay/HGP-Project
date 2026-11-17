"""
HICP: horizon-indexed causal propagation with causal, physical and semantic branches.

    Z       = Encoder(X)                              per-node GRU, shared weights, no cross-sensor mixing
    S_h^c   = GELU(G_h Z W_h^c + b_h^c)               one directed hop over the horizon-h causal graph (Part A)
    S_h^p   = GELU(A_road Z W_h^p + b_h^p)            one hop over the road adjacency (shared graph, per-horizon weights)
    S_h^s   = GELU(A_sem Z W_h^s + b_h^s)             one hop over a learned similarity graph (shared graph, per-horizon weights)
    A_sem   = softmax_j(E_t E_s^T / sqrt(d_s))        top-k sources per target, learned node embeddings E_t, E_s, no self-loops
    alpha_h = softmax([0, MLP([Z, S_h^c, S_h^p, S_h^s, e_h])])   one weight per branch, per node and feature;
                                                                  the self branch has a fixed zero logit
    H_h     = alpha_h^0 Z + sum_m alpha_h^m S_h^m       convex mixture of the self path and the three branches
    y_h     = H_h w_h + b_h                             one head per horizon, all horizons in one forward pass
"""

import math

import torch
from torch import nn

from .config import ModelConfig

BRANCHES = ("causal", "physical", "semantic")


def propagate(graph: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """graph: [N, N] (sparse or dense) with graph[i, j] the weight of source j for target i; Z: [B, N, d]."""
    B, N, d = Z.shape
    flat = Z.permute(1, 0, 2).reshape(N, B * d)
    out = torch.sparse.mm(graph, flat) if graph.is_sparse else graph @ flat
    return out.reshape(N, B, d).permute(1, 0, 2)


class Encoder(nn.Module):
    """GRU shared by all sensors; each sensor sees only its own history (speed + 4 time features)."""

    def __init__(self, num_features: int, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        B, T, N = flow.shape
        x = torch.cat([flow.unsqueeze(-1), time_feat.unsqueeze(2).expand(B, T, N, -1)], -1)
        _, h = self.gru(x.permute(0, 2, 1, 3).reshape(B * N, T, -1))
        return self.dropout(h[-1]).reshape(B, N, -1)


class Propagation(nn.Module):
    """One directed hop over a graph: GELU(A Z W + b)."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden)])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Z: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.layers[0](propagate(graph, Z))))


class SemanticGraph(nn.Module):
    """Learned static similarity graph: A[i, j] = softmax_j(E_t[i] . E_s[j] / sqrt(d)) over the top_k most similar sources j != i."""

    def __init__(self, num_nodes: int, dim: int, top_k: int):
        super().__init__()
        self.target = nn.Parameter(torch.randn(num_nodes, dim) / math.sqrt(dim))
        self.source = nn.Parameter(torch.randn(num_nodes, dim) / math.sqrt(dim))
        self.top_k = min(top_k, num_nodes - 1)

    def forward(self) -> torch.Tensor:
        scores = (self.target @ self.source.T) / math.sqrt(self.target.shape[1])
        scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool, device=scores.device), float("-inf"))
        if self.top_k < len(scores) - 1:
            threshold = scores.topk(self.top_k, dim=1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        return torch.softmax(scores, dim=1)


class Gate(nn.Module):
    """Softmax over [self, causal, physical, semantic] per node and feature; the self logit is fixed at zero.
    The horizon enters only through its learned embedding e_h; the MLP is shared across horizons."""

    def __init__(self, hidden: int, num_horizons: int, embed: int, branches: int = len(BRANCHES)):
        super().__init__()
        self.branches = branches
        self.embed = nn.Embedding(num_horizons, embed)
        self.mlp = nn.Sequential(nn.Linear((1 + branches) * hidden + embed, hidden), nn.GELU(), nn.Linear(hidden, branches * hidden))

    def forward(self, Z: torch.Tensor, S: list, k: int) -> torch.Tensor:
        """Returns alpha [B, N, 1 + branches, d]; alpha[..., 0, :] weights Z, alpha[..., m, :] weights S[m - 1]."""
        B, N, d = Z.shape
        e = self.embed.weight[k].expand(B, N, -1)
        logits = self.mlp(torch.cat([Z, *S, e], -1)).view(B, N, self.branches, d)
        logits = torch.cat([logits.new_zeros(B, N, 1, d), logits], 2)
        return torch.softmax(logits, dim=2)


class HICP(nn.Module):
    def __init__(self, num_features: int, horizons: tuple, config: ModelConfig, num_nodes: int):
        super().__init__()
        self.horizons = tuple(horizons)
        self.config = config
        self.branches = list(BRANCHES)
        self.encoder = Encoder(num_features, config.hidden, config.dropout)
        self.propagation = nn.ModuleList([Propagation(config.hidden, config.dropout) for _ in horizons])
        self.physical_propagation = nn.ModuleList([Propagation(config.hidden, config.dropout) for _ in horizons])
        self.semantic_graph = SemanticGraph(num_nodes, config.semantic_dim, config.semantic_top_k)
        self.semantic_propagation = nn.ModuleList([Propagation(config.hidden, config.dropout) for _ in horizons])
        self.gate = Gate(config.hidden, len(horizons), config.horizon_embed, len(self.branches))
        self.heads = nn.ModuleList([nn.Linear(config.hidden, 1) for _ in horizons])

    def forward(self, flow: torch.Tensor, time_feat: torch.Tensor, graphs: dict, physical: torch.Tensor):
        """
        flow [B, T, N] standardised inputs, time_feat [B, T, 4]; graphs: the row-normalised causal graph G_h per horizon;
        physical: the row-normalised road adjacency.
        Returns predictions [B, H, N] (standardised) and, per horizon, the mean gate weight of every branch.
        """
        Z = self.encoder(flow, time_feat)
        semantic = self.semantic_graph()
        preds, gates = [], {}
        for k, h in enumerate(self.horizons):
            S = [self.propagation[k](Z, graphs[h]),
                 self.physical_propagation[k](Z, physical),
                 self.semantic_propagation[k](Z, semantic)]
            alpha = self.gate(Z, S, k)
            H = alpha[:, :, 0] * Z + sum(alpha[:, :, m + 1] * S[m] for m in range(len(S)))
            preds.append(self.heads[k](H).squeeze(-1))
            gates[h] = {b: alpha[:, :, m + 1].detach().mean() for m, b in enumerate(self.branches)}
        return torch.stack(preds, 1), gates


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
