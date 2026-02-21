from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from omniflow_l1.agent.core import AgentConfig, OmniFlowL1Agent
from omniflow_l1.critic.physics import CriticConfig, PhysicsCritic
from omniflow_l1.data.datasets import DatasetConfig, iter_sequences, load_benchmark_data
from omniflow_l1.eval.metrics import psnr, rmse, ssim_mean
from omniflow_l1.nes.simulator import NESConfig, PerturbativePersistenceNES
from omniflow_l1.rag.retriever import HierarchicalRetriever, RAGConfig
from omniflow_l1.symbolic.projector import SymbolicConfig, VisualSymbolicProjector
from omniflow_l1.utils import ensure_dir, load_and_merge_yaml, load_yaml, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[], help="Can be passed multiple times.")
    parser.add_argument("--config-list", type=str, default="", help="Comma-separated config files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs/data without inference.")
    return parser.parse_args()


def _resolve_cfg(args: argparse.Namespace) -> dict:
    paths = list(args.config)
    if args.config_list:
        paths.extend([x.strip() for x in args.config_list.split(",") if x.strip()])
    if not paths:
        raise ValueError("At least one --config is required.")
    if len(paths) == 1:
        return load_yaml(paths[0])
    return load_and_merge_yaml(paths)


def main() -> None:
    args = parse_args()
    cfg = _resolve_cfg(args)
    set_seed(int(cfg["seed"]))

    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "reports"))

    dcfg = DatasetConfig(
        sequence_length=cfg["data"]["sequence_length"],
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        channels=cfg["data"]["channels"],
        samples=cfg["data"]["samples"],
        data_root=cfg["data"].get("data_root", "data"),
        use_synthetic_if_missing=bool(cfg["data"].get("use_synthetic_if_missing", True)),
    )
    data = load_benchmark_data(cfg["data"]["benchmark"], dcfg)
    if args.dry_run:
        expected = Path(dcfg.data_root) / cfg["data"]["benchmark"] / "dataset.npy"
        print(
            {
                "status": "ok",
                "dry_run": True,
                "benchmark": cfg["data"]["benchmark"],
                "expected_dataset": str(expected),
                "synthetic_fallback": dcfg.use_synthetic_if_missing,
                "loaded_shape": list(data.shape),
            }
        )
        return

    nes = PerturbativePersistenceNES(
        NESConfig(
            ensemble_k=int(cfg["nes"]["ensemble_k"]),
            perturbation_lambda=float(cfg["nes"]["perturbation_lambda"]),
        )
    )
    projector = VisualSymbolicProjector(SymbolicConfig(topk_regions=int(cfg["symbolic"]["topk_regions"])))
    critic = PhysicsCritic(
        CriticConfig(
            divergence_threshold=float(cfg["critic"]["divergence_threshold"]),
            energy_tolerance=float(cfg["critic"]["energy_tolerance"]),
        )
    )
    retriever = HierarchicalRetriever(
        RAGConfig(top_k=int(cfg["rag"]["top_k"])),
        knowledge=cfg["rag"]["knowledge"],
    )
    agent = OmniFlowL1Agent(
        AgentConfig(
            uncertainty_threshold=float(cfg["agent"]["uncertainty_threshold"]),
            enable_counterfactual=bool(cfg["agent"]["enable_counterfactual"]),
        ),
        nes=nes,
        projector=projector,
        critic=critic,
        retriever=retriever,
    )

    all_pred = []
    all_gt = []
    all_reports = []
    train_steps = int(cfg["data"]["train_steps"])
    test_steps = int(cfg["data"]["test_steps"])

    for idx, (x_init, x_target) in enumerate(tqdm(iter_sequences(data, train_steps, test_steps), desc="L1 run")):
        result = agent.run(
            x_init=x_init,
            steps=test_steps,
            instruction="Generate a physically grounded forecast and concise analysis.",
        )
        pred = result["pred_mean"]
        all_pred.append(pred)
        all_gt.append(x_target)
        all_reports.append(result["report"])

        report_path = os.path.join(out_dir, "reports", f"report_{idx:03d}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(result["report"])

    pred_np = np.asarray(all_pred, dtype=np.float32)
    gt_np = np.asarray(all_gt, dtype=np.float32)
    if pred_np.size == 0:
        raise RuntimeError("No valid sequences generated. Check dataset shape and step configuration.")
    np.save(os.path.join(out_dir, "predictions.npy"), pred_np)
    np.save(os.path.join(out_dir, "ground_truth.npy"), gt_np)

    metrics = {
        "rmse": rmse(gt_np, pred_np),
        "ssim": ssim_mean(gt_np[0], pred_np[0]),
        "psnr": psnr(gt_np, pred_np),
        "num_sequences": int(pred_np.shape[0]),
    }
    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
