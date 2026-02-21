from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NESConfig:
    ensemble_k: int
    perturbation_lambda: float


class PerturbativePersistenceNES:
    """
    L1 NES baseline:
    - Uses perturbative latent-like noise injection to build an ensemble
    - Evolves states with a simple persistence+advection proxy
    """

    def __init__(self, cfg: NESConfig):
        self.cfg = cfg

    def _rollout_once(self, x0: np.ndarray, steps: int, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
        state = x0.copy()
        out = []
        for _ in range(steps):
            drift = 0.99 * np.roll(state, shift=1, axis=-1) + 0.01 * np.roll(state, shift=1, axis=-2)
            state = drift + rng.normal(0.0, noise_scale, size=state.shape).astype(np.float32)
            out.append(state.copy())
        return np.stack(out, axis=0)

    def forecast_ensemble(self, x_init: np.ndarray, steps: int) -> np.ndarray:
        # x_init expected shape: [T0, C, H, W], use the final frame as initial condition.
        x0 = x_init[-1]
        ensemble = []
        for k in range(self.cfg.ensemble_k):
            rng = np.random.default_rng(seed=10_000 + k)
            perturbed = x0 + rng.normal(0.0, self.cfg.perturbation_lambda, size=x0.shape).astype(np.float32)
            rollout = self._rollout_once(perturbed, steps=steps, noise_scale=self.cfg.perturbation_lambda, rng=rng)
            ensemble.append(rollout)
        return np.stack(ensemble, axis=0)  # [K, T, C, H, W]

    @staticmethod
    def ensemble_mean_std(ensemble_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = ensemble_pred.mean(axis=0)
        std = ensemble_pred.std(axis=0)
        return mean, std
