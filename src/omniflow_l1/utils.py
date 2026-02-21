from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class Batch:
    x_init: np.ndarray
    x_target: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_and_merge_yaml(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        cfg = load_yaml(path) or {}
        merged = deep_merge(merged, cfg)
    return merged


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
