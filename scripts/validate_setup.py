from __future__ import annotations

import argparse
from pathlib import Path

from omniflow_l1.utils import load_and_merge_yaml


REQUIRED_TOP_LEVEL = ["seed", "output_dir", "data", "nes", "symbolic", "critic", "rag", "agent"]
REQUIRED_DATA_KEYS = ["benchmark", "data_root", "sequence_length", "height", "width", "channels"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Pass multiple times for layered config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_and_merge_yaml(args.config)

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    dcfg = cfg["data"]
    missing_data = [k for k in REQUIRED_DATA_KEYS if k not in dcfg]
    if missing_data:
        raise ValueError(f"Missing data config keys: {missing_data}")

    dataset_file = Path(dcfg["data_root"]) / dcfg["benchmark"] / "dataset.npy"
    can_fallback = bool(dcfg.get("use_synthetic_if_missing", False))

    print(
        {
            "status": "ok",
            "benchmark": dcfg["benchmark"],
            "dataset_expected": str(dataset_file),
            "dataset_exists": dataset_file.exists(),
            "synthetic_fallback": can_fallback,
            "config_chain": args.config,
        }
    )


if __name__ == "__main__":
    main()
