import numpy as np
import torch


def to_device(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    return inputs.to(device, non_blocking=True)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1 (for CrossEntropyLoss's `weight` arg)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
