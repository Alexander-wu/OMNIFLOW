from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SymbolicConfig:
    topk_regions: int


class VisualSymbolicProjector:
    """
    Converts flow tensors into compact symbolic descriptors
    that the agent can reason over.
    """

    def __init__(self, cfg: SymbolicConfig):
        self.cfg = cfg

    def encode(self, x: np.ndarray) -> dict[str, object]:
        # x shape [T, C, H, W]
        t, c, h, w = x.shape
        stats = {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "max": float(x.max()),
            "min": float(x.min()),
            "shape": [t, c, h, w],
        }

        # Simple topological proxy: strongest-gradient regions.
        final = x[-1].mean(axis=0)
        gy, gx = np.gradient(final)
        grad_mag = np.sqrt(gx**2 + gy**2)
        flat_idx = np.argsort(grad_mag.reshape(-1))[-self.cfg.topk_regions :]
        regions = []
        for idx in flat_idx[::-1]:
            yy, xx = divmod(int(idx), w)
            regions.append({"y": int(yy), "x": int(xx), "strength": float(grad_mag[yy, xx])})

        return {
            "global_stats": stats,
            "salient_regions": regions,
            "descriptors": self._make_descriptors(stats, regions),
        }

    @staticmethod
    def _make_descriptors(stats: dict[str, float], regions: list[dict[str, float]]) -> list[str]:
        tokens = []
        if stats["std"] > 0.2:
            tokens.append("high_variability")
        else:
            tokens.append("moderate_variability")
        if stats["max"] - stats["min"] > 1.0:
            tokens.append("sharp_gradient_structure")
        if regions:
            tokens.append("localized_dynamic_hotspots")
        return tokens
