from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class DatasetConfig:
    sequence_length: int
    height: int
    width: int
    channels: int
    samples: int


def _make_synthetic_flow(cfg: DatasetConfig, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros(
        (cfg.samples, cfg.sequence_length, cfg.channels, cfg.height, cfg.width),
        dtype=np.float32,
    )
    for i in range(cfg.samples):
        x = rng.standard_normal((cfg.channels, cfg.height, cfg.width)).astype(np.float32) * 0.1
        for t in range(cfg.sequence_length):
            noise = rng.standard_normal(x.shape).astype(np.float32) * 0.02
            x = 0.98 * np.roll(x, shift=1, axis=-1) + 0.02 * np.roll(x, shift=1, axis=-2) + noise
            data[i, t] = x
    return data


def load_benchmark_data(benchmark: str, cfg: DatasetConfig) -> np.ndarray:
    # L1 fallback: synthetic proxy that preserves spatiotemporal tensor shape.
    # Real benchmark loaders can be plugged here with the same output format.
    if benchmark not in {"turbulence2d", "era5", "sevir"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return _make_synthetic_flow(cfg)


def iter_sequences(data: np.ndarray, train_steps: int, test_steps: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    total_steps = train_steps + test_steps
    for item in data:
        if item.shape[0] < total_steps + 1:
            continue
        x_init = item[:train_steps]
        x_target = item[train_steps : train_steps + test_steps]
        yield x_init, x_target
