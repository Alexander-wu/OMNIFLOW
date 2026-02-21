from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class DatasetConfig:
    sequence_length: int
    height: int
    width: int
    channels: int
    samples: int
    data_root: str = "data"
    use_synthetic_if_missing: bool = True


def _load_npy_dataset(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D tensor [N,T,C,H,W], got shape={arr.shape} from {path}")
    return arr.astype(np.float32)


def _expected_path(benchmark: str, data_root: str) -> Path:
    return Path(data_root) / benchmark / "dataset.npy"


def _validate_shape(data: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    _, t, c, h, w = data.shape
    if t < cfg.sequence_length:
        raise ValueError(
            f"Dataset sequence too short: got {t}, requires at least {cfg.sequence_length}. "
            "Adjust config.data.sequence_length or provide longer sequences."
        )
    if c < cfg.channels:
        raise ValueError(
            f"Dataset channels too few: got {c}, requires at least {cfg.channels}."
        )
    if h != cfg.height or w != cfg.width:
        raise ValueError(
            f"Resolution mismatch: data=({h},{w}) vs config=({cfg.height},{cfg.width})."
        )
    return data[:, : cfg.sequence_length, : cfg.channels, :, :]


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
    if benchmark not in {"turbulence2d", "era5", "sevir"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    expected = _expected_path(benchmark, cfg.data_root)
    loaded = _load_npy_dataset(expected)
    if loaded is not None:
        return _validate_shape(loaded, cfg)
    if cfg.use_synthetic_if_missing:
        return _make_synthetic_flow(cfg)
    raise FileNotFoundError(
        f"Missing dataset file: {expected}. "
        "Create it as [N,T,C,H,W] float array or enable synthetic fallback."
    )


def iter_sequences(data: np.ndarray, train_steps: int, test_steps: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    total_steps = train_steps + test_steps
    for item in data:
        if item.shape[0] < total_steps:
            continue
        x_init = item[:train_steps]
        x_target = item[train_steps : train_steps + test_steps]
        yield x_init, x_target
