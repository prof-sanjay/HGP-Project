import numpy as np
import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Training loss. pred, target [B, H, N]; mask [B, H]. Horizons are averaged with equal weight."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    err = ((pred - torch.nan_to_num(target)).abs() * m).sum((0, 2))
    count = (m.sum((0, 2)) * pred.shape[-1]).clamp_min(1.0)
    present = (mask.sum(0) > 0).to(pred.dtype)
    return ((err / count) * present).sum() / present.sum().clamp_min(1.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, mape_min_flow: float = 1.0) -> dict:
    """MAE, RMSE and MAPE (over targets of at least mape_min_flow) on observed targets."""
    y_true, y_pred = np.asarray(y_true, np.float64), np.asarray(y_pred, np.float64)
    err = y_pred - y_true
    keep = np.abs(y_true) >= mape_min_flow
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(100 * np.mean(np.abs(err[keep]) / np.abs(y_true[keep]))) if keep.any() else float("nan"),
        "n": int(y_true.shape[0]),
    }


def horizon_metrics(targets: dict, predictions: dict, mape_min_flow: float = 1.0) -> dict:
    """Metrics per horizon and their mean over the horizons."""
    out = {int(h): evaluate(targets[h], predictions[h], mape_min_flow) for h in sorted(targets)}
    hs = sorted(out)
    out["mean"] = {k: float(np.mean([out[h][k] for h in hs])) for k in ("mae", "rmse", "mape")}
    return out
