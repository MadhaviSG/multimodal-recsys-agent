"""
Master training script — runs full RecSys training pipeline in order.

Usage:
    python scripts/train_recsys.py --config configs/config.yaml

Order:
    1. Preprocess (if not already done)
    2. Train Mult-VAE
    3. Train Two-Tower
    4. Build Qdrant index
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def run(cmd: str):
    print(f"\n{'='*55}")
    print(f"Running: {cmd}")
    print('='*55)
    result = subprocess.run(cmd, shell=True, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_mult_vae", action="store_true")
    parser.add_argument("--skip_two_tower", action="store_true")
    parser.add_argument("--skip_index", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Step 1: Preprocess
    if not args.skip_preprocess:
        if not Path("data/processed/train.npz").exists():
            run(f"python scripts/download_data.py")
            run(f"python scripts/preprocess.py --config {args.config}")
        else:
            print("Processed data already exists — skipping preprocessing.")

    # Step 2: Train Mult-VAE
    if not args.skip_mult_vae:
        run(f"python scripts/train_mult_vae.py --config {args.config}")

    # Step 3: Train Two-Tower
    if not args.skip_two_tower:
        run(f"python scripts/train_two_tower.py --config {args.config}")

    # Step 4: Build Qdrant index
    if not args.skip_index:
        run(f"python scripts/build_index.py --config {args.config}")

    print("\n✓ RecSys pipeline complete.")
    print("Run: python scripts/run_agent.py to start the agent.")


if __name__ == "__main__":
    main()