from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root
    os.makedirs(root, exist_ok=True)
    for name in ["turbulence2d", "era5", "sevir"]:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        note = os.path.join(d, "README.txt")
        if not os.path.exists(note):
            with open(note, "w", encoding="utf-8") as f:
                f.write(
                    "Place raw dataset files here.\n"
                    f"Dataset: {name}\n"
                    "This L1 scaffold currently uses synthetic fallback when real data is absent.\n"
                )
    print(f"Prepared data directories under: {root}")


if __name__ == "__main__":
    main()
