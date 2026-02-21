from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    data_range = float(np.max(y_true) - np.min(y_true) + 1e-6)
    return float(peak_signal_noise_ratio(y_true, y_pred, data_range=data_range))


def ssim_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Input shape [T, C, H, W]
    vals = []
    for t in range(y_true.shape[0]):
        for c in range(y_true.shape[1]):
            vals.append(
                structural_similarity(
                    y_true[t, c],
                    y_pred[t, c],
                    data_range=float(y_true[t, c].max() - y_true[t, c].min() + 1e-6),
                )
            )
    return float(np.mean(vals))
