from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from omniflow_l1.critic.physics import PhysicsCritic
from omniflow_l1.nes.simulator import PerturbativePersistenceNES
from omniflow_l1.rag.retriever import HierarchicalRetriever
from omniflow_l1.symbolic.projector import VisualSymbolicProjector


@dataclass
class AgentConfig:
    uncertainty_threshold: float
    enable_counterfactual: bool


class OmniFlowL1Agent:
    def __init__(
        self,
        cfg: AgentConfig,
        nes: PerturbativePersistenceNES,
        projector: VisualSymbolicProjector,
        critic: PhysicsCritic,
        retriever: HierarchicalRetriever,
    ) -> None:
        self.cfg = cfg
        self.nes = nes
        self.projector = projector
        self.critic = critic
        self.retriever = retriever

    def run(self, x_init: np.ndarray, steps: int, instruction: str) -> dict[str, object]:
        # ReAct-like step 1: simulate
        ensemble = self.nes.forecast_ensemble(x_init=x_init, steps=steps)
        mean_pred, std_pred = self.nes.ensemble_mean_std(ensemble)
        sigma_ens = float(std_pred.mean())

        # ReAct-like step 2: symbolic projection + retrieval
        symbolic = self.projector.encode(mean_pred)
        retrieval = self.retriever.retrieve(instruction)

        # ReAct-like step 3: consistency check
        check = self.critic.check(x_pred=mean_pred, x_init=x_init)

        counterfactual = None
        if self.cfg.enable_counterfactual and sigma_ens > self.cfg.uncertainty_threshold:
            # Simplified counterfactual: damp first channel forcing proxy.
            x_cf = x_init.copy()
            x_cf[:, 0] *= 0.5
            ens_cf = self.nes.forecast_ensemble(x_init=x_cf, steps=steps)
            mean_cf, _ = self.nes.ensemble_mean_std(ens_cf)
            delta = float((mean_cf - mean_pred).mean())
            counterfactual = {
                "triggered": True,
                "sigma_ens": sigma_ens,
                "mean_delta": delta,
            }
        else:
            counterfactual = {"triggered": False, "sigma_ens": sigma_ens}

        report = self._build_report(symbolic, retrieval, check, counterfactual)
        return {
            "pred_mean": mean_pred,
            "pred_std": std_pred,
            "consistency": check,
            "counterfactual": counterfactual,
            "symbolic": symbolic,
            "retrieval": retrieval,
            "report": report,
        }

    @staticmethod
    def _build_report(
        symbolic: dict[str, object],
        retrieval: dict[str, list[str]],
        check: dict[str, object],
        counterfactual: dict[str, object],
    ) -> str:
        now = datetime.utcnow().isoformat(timespec="seconds")
        lines = [
            f"# OMNIFLOW L1 Scientific Report ({now}Z)",
            "",
            "## Executive Summary",
            f"- Descriptor tokens: {', '.join(symbolic['descriptors'])}",
            f"- Consistency passed: {check['passed']}",
            f"- Counterfactual triggered: {counterfactual['triggered']}",
            "",
            "## Statistical Overview",
            f"- Mean: {symbolic['global_stats']['mean']:.4f}",
            f"- Std: {symbolic['global_stats']['std']:.4f}",
            f"- Min/Max: {symbolic['global_stats']['min']:.4f} / {symbolic['global_stats']['max']:.4f}",
            "",
            "## Physical Consistency",
            f"- Divergence score: {check['divergence_score']:.4f}",
            f"- Energy shift ratio: {check['energy_shift_ratio']:.4f}",
            f"- Violations: {', '.join(check['violations']) if check['violations'] else 'None'}",
            "",
            "## Retrieved Knowledge",
            f"- Kphy: {retrieval.get('kphy', [])}",
            f"- Kprot: {retrieval.get('kprot', [])}",
            f"- Khist: {retrieval.get('khist', [])}",
            "",
            "## Conclusion",
            "- This is an L1 reproducible pipeline report with physics checks and auditable reasoning traces.",
        ]
        return "\n".join(lines)
