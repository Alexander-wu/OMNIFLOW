from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CriticConfig:
    divergence_threshold: float
    energy_tolerance: float


class PhysicsCritic:
    def __init__(self, cfg: CriticConfig):
        self.cfg = cfg

    @staticmethod
    def _divergence_2d(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        dvx_dx = np.gradient(vx, axis=-1)
        dvy_dy = np.gradient(vy, axis=-2)
        return dvx_dx + dvy_dy

    def check(self, x_pred: np.ndarray, x_init: np.ndarray) -> dict[str, object]:
        # x_pred: [T, C, H, W], x_init: [T0, C, H, W]
        # If channels >=2 treat first two as velocity components.
        if x_pred.shape[1] >= 2:
            div = self._divergence_2d(x_pred[-1, 0], x_pred[-1, 1])
            div_score = float(np.abs(div).mean())
        else:
            div_score = 0.0

        e0 = float((x_init[-1] ** 2).mean())
        e1 = float((x_pred[-1] ** 2).mean())
        energy_shift = abs(e1 - e0) / (abs(e0) + 1e-6)

        ok_div = div_score <= self.cfg.divergence_threshold
        ok_energy = energy_shift <= self.cfg.energy_tolerance
        passed = bool(ok_div and ok_energy)

        return {
            "passed": passed,
            "divergence_score": div_score,
            "energy_shift_ratio": float(energy_shift),
            "violations": [
                name
                for name, flag in [
                    ("mass_conservation", ok_div),
                    ("energy_stability", ok_energy),
                ]
                if not flag
            ],
        }
