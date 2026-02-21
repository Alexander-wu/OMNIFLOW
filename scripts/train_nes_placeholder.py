from __future__ import annotations

import argparse

from omniflow_l1.utils import load_and_merge_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly allow training execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_and_merge_yaml(args.config)
    train_cfg = cfg.get("train", {})
    if not args.execute:
        print(
            {
                "status": "template_only",
                "message": "Training is skipped by default. Pass --execute when data is ready.",
                "epochs": train_cfg.get("epochs", 0),
                "batch_size": train_cfg.get("batch_size", 0),
                "learning_rate": train_cfg.get("learning_rate", 0.0),
            }
        )
        return

    raise NotImplementedError(
        "Placeholder script: integrate your stronger NES training loop here when data is prepared."
    )


if __name__ == "__main__":
    main()
