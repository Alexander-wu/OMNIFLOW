from __future__ import annotations

import argparse
import json

import numpy as np

from omniflow_l1.eval.metrics import psnr, rmse, ssim_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="outputs/predictions.npy")
    parser.add_argument("--gt", type=str, default="outputs/ground_truth.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = np.load(args.pred)
    gt = np.load(args.gt)
    result = {
        "rmse": rmse(gt, pred),
        "ssim": ssim_mean(gt[0], pred[0]),
        "psnr": psnr(gt, pred),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
