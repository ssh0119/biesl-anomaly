import numpy as np
import wfdb

from src import config


def load_channels(folder_path: str, channels: list[str]) -> dict[str, np.ndarray]:
    """Reads a WFDB record and returns {channel_name: float32 array of length SEGMENT_LEN}."""
    record_path = str(config.DATA_ROOT / folder_path)
    record = wfdb.rdrecord(record_path)
    sig_name = record.sig_name
    signal = record.p_signal  # (n_samples, n_channels), float64, NaN where signal is missing/invalid

    out = {}
    for ch in channels:
        if ch not in sig_name:
            raise ValueError(f"channel {ch!r} not present in {folder_path} (has {sig_name})")
        col = signal[:, sig_name.index(ch)].astype(np.float32)
        if col.shape[0] != config.SEGMENT_LEN:
            fixed = np.zeros(config.SEGMENT_LEN, dtype=np.float32)
            n = min(config.SEGMENT_LEN, col.shape[0])
            fixed[:n] = col[:n]
            col = fixed
        out[ch] = col
    return out


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-segment z-score normalization; NaNs (dropout in the recording) are filled with 0 (post-normalization mean)."""
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    mean = x[finite].mean()
    std = x[finite].std()
    std = std if std > 1e-6 else 1.0
    x = (x - mean) / std
    x = np.where(finite, x, 0.0)
    return x.astype(np.float32)
