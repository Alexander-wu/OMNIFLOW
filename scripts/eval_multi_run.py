from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", type=str, default="outputs/**/metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(".").glob(args.metrics_glob))
    if not files:
        print({"status": "no_files", "pattern": args.metrics_glob})
        return

    rmse_vals, ssim_vals, psnr_vals = [], [], []
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        if "rmse" in payload:
            rmse_vals.append(float(payload["rmse"]))
        if "ssim" in payload:
            ssim_vals.append(float(payload["ssim"]))
        if "psnr" in payload:
            psnr_vals.append(float(payload["psnr"]))

    def stats(x: list[float]) -> dict[str, float]:
        arr = np.asarray(x, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}

    result = {
        "files": [str(f) for f in files],
        "rmse": stats(rmse_vals) if rmse_vals else {},
        "ssim": stats(ssim_vals) if ssim_vals else {},
        "psnr": stats(psnr_vals) if psnr_vals else {},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
