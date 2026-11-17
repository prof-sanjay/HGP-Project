"""Settings of the proposed model. Every value here is the one used in the paper."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """PEMS-BAY: chronological 70 / 10 / 20 split, 12 input steps, horizons of 15, 30 and 60 minutes."""
    path: str = "data/pems_bay.npz"
    train_rate: float = 0.7
    val_rate: float = 0.1
    seq_len: int = 12
    horizons: tuple = (3, 6, 12)
    mape_min_flow: float = 1.0     # MAPE counts targets of at least 1 mph
    min_std: float = 0.1           # smallest per-sensor standard deviation used for scaling


@dataclass
class GraphConfig:
    """Part A: two-hop ridge VAR, impulse responses, group-knockoff selection, graph assembly."""
    var_lag: int = 3
    var_ridge: float = 1e6
    stride: int = 3                # one row every 15 minutes for the knockoff statistic
    self_lags: int = 3             # own lags projected out of the response and the candidates
    shrink: float = 0.01           # ridge shrinkage of the candidate correlation matrix
    group_corr: float = 0.5        # candidates with |r| >= group_corr form one knockoff group
    fdr: float = 0.3               # knockoff+ level q
    max_sources: int = 25          # sources kept per target
    min_rows: int = 2000           # fewest usable rows for a target's knockoff statistic
    out_dir: str = "graphs"


@dataclass
class ModelConfig:
    """Part B: HICP with the causal, physical (road) and semantic (learned similarity) branches."""
    hidden: int = 128
    horizon_embed: int = 8
    dropout: float = 0.1
    semantic_dim: int = 16         # node-embedding size of the semantic graph
    semantic_top_k: int = 25       # sources kept per target in the semantic graph


@dataclass
class TrainConfig:
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 40
    warmup_epochs: int = 1
    patience: int = 6
    grad_clip: float = 5.0
    results_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)

    def run_dir(self) -> Path:
        return Path(self.results_dir) / f"proposed__seed{self.seed}"
