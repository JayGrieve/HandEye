#!/usr/bin/env python3
"""
Download DINOv3 Small (ViT-S/16) model to local directory.

Usage:
    HF_TOKEN=your_token python download_dinov3.py

Note: This model is gated and requires accepting the license at:
    https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
"""

import os
from pathlib import Path

def main():
    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    save_dir = Path(__file__).parent / "models" / "dinov3-small"

    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Usage: HF_TOKEN=your_token python download_dinov3.py")
        print("\nNote: You must also accept the model license at:")
        print(f"  https://huggingface.co/{model_id}")
        return

    print(f"Downloading {model_id}...")
    print(f"Saving to: {save_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Download using huggingface_hub
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        token=token,
    )

    print(f"\nDownload complete!")
    print(f"Model saved to: {save_dir}")

    # Show downloaded files
    print("\nDownloaded files:")
    for f in sorted(save_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(save_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
