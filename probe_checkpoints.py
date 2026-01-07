#!/usr/bin/env python3
"""
Evaluate multiple MAE checkpoints via linear probing.

Runs linear probing on checkpoints at different training epochs to track
how representation quality evolves during continued pretraining.

Usage:
    python probe_checkpoints.py --checkpoint_dir ./mae_output_imagenet_init --data_dir ./epic_kitchens
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_checkpoints(checkpoint_dir: Path, target_epochs: list[int]) -> dict[int, Path]:
    """Find checkpoint files closest to target epochs."""
    checkpoints = {}

    # List all checkpoint files
    all_ckpts = list(checkpoint_dir.glob("checkpoint-*.pth"))

    # Parse epoch numbers (format: checkpoint-0009.pth)
    epoch_to_path = {}
    for ckpt in all_ckpts:
        name = ckpt.stem  # checkpoint-0009
        if name == "checkpoint-latest":
            continue
        try:
            epoch = int(name.split("-")[1])
            epoch_to_path[epoch] = ckpt
        except (IndexError, ValueError):
            continue

    available_epochs = sorted(epoch_to_path.keys())
    print(f"Available checkpoints: {available_epochs}")

    # Find closest checkpoint for each target
    for target in target_epochs:
        if target in epoch_to_path:
            checkpoints[target] = epoch_to_path[target]
        else:
            # Find closest available epoch
            closest = min(available_epochs, key=lambda x: abs(x - target), default=None)
            if closest is not None:
                checkpoints[target] = epoch_to_path[closest]
                if closest != target:
                    print(f"  Target epoch {target} not found, using closest: {closest}")

    return checkpoints


def run_probe(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    probe_epochs: int = 50,
    batch_size: int = 256,
    model: str = "vit_base_patch16",
) -> dict:
    """Run linear probing on a single checkpoint."""

    cmd = [
        sys.executable,
        "linear_probe.py",
        "--checkpoint", str(checkpoint_path),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--epochs", str(probe_epochs),
        "--batch_size", str(batch_size),
        "--model", model,
        "--skip_random",
        "--skip_imagenet",
    ]

    print(f"\nRunning: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running probe:")
        print(result.stderr)
        return None

    print(result.stdout)

    # Parse results from output
    lines = result.stdout.split("\n")
    for line in lines:
        if "Final:" in line:
            # Extract accuracy: "Final: val_acc=34.76% (best=34.76%)"
            try:
                best_acc = float(line.split("best=")[1].split("%")[0])
                return {"best_val_acc": best_acc}
            except:
                pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints via linear probing")
    parser.add_argument("--checkpoint_dir", type=Path, required=True,
                        help="Directory containing MAE checkpoints")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to epic_kitchens directory")
    parser.add_argument("--output_dir", type=Path, default="./probe_results",
                        help="Base output directory for results")
    parser.add_argument("--epochs", type=int, nargs="+", default=[9, 29, 49, 79, 99, 129],
                        help="Target checkpoint epochs to evaluate (default: 9 29 49 79 99 129)")
    parser.add_argument("--probe_epochs", type=int, default=50,
                        help="Number of epochs for linear probing (default: 50)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for probing")
    parser.add_argument("--model", type=str, default="vit_base_patch16",
                        help="Model architecture")
    args = parser.parse_args()

    # Create unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"checkpoint_sweep_{timestamp}"
    exp_dir = args.output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output directory: {exp_dir}")

    # Save experiment config
    config = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "data_dir": str(args.data_dir),
        "target_epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "timestamp": timestamp,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoint_dir, args.epochs)

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"\nWill evaluate {len(checkpoints)} checkpoints:")
    for epoch, path in sorted(checkpoints.items()):
        print(f"  Epoch {epoch}: {path.name}")

    # Run probing for each checkpoint
    results = {}

    for epoch, ckpt_path in sorted(checkpoints.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating epoch {epoch}: {ckpt_path.name}")
        print(f"{'='*60}")

        # Create subdirectory for this checkpoint's probe results
        probe_output = exp_dir / f"epoch_{epoch:04d}"
        probe_output.mkdir(exist_ok=True)

        result = run_probe(
            ckpt_path,
            args.data_dir,
            probe_output,
            args.probe_epochs,
            args.batch_size,
            args.model,
        )

        if result:
            results[epoch] = result["best_val_acc"]
            print(f"Epoch {epoch}: {result['best_val_acc']:.2f}%")
        else:
            results[epoch] = None
            print(f"Epoch {epoch}: FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Linear Probe Accuracy vs Pretraining Epoch")
    print(f"{'='*60}")
    print(f"{'Epoch':>10} | {'Val Acc':>10}")
    print("-" * 25)
    for epoch in sorted(results.keys()):
        acc = results[epoch]
        if acc is not None:
            print(f"{epoch:>10} | {acc:>9.2f}%")
        else:
            print(f"{epoch:>10} | {'FAILED':>10}")

    # Save summary
    summary = {
        "experiment": exp_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "probe_epochs": args.probe_epochs,
        "results": {str(k): v for k, v in results.items()},
    }
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {exp_dir}")

    # Also append to a global log file
    log_file = args.output_dir / "checkpoint_sweep_log.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Appended to {log_file}")


if __name__ == "__main__":
    main()
