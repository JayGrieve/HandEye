#!/usr/bin/env python3
"""
Linear probing evaluation for MAE pretrained models on EPIC-KITCHENS.

Freezes the encoder and trains a linear classifier on top.
Compares against random initialization and ImageNet pretrained weights.

Usage:
    python linear_probe.py --checkpoint ./mae_output/checkpoint-0369.pth --data_dir ./epic_kitchens

Results are saved to ./probe_results/<experiment_name>/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Add MAE to path
sys.path.insert(0, str(Path(__file__).parent / "mae"))
import models_vit


class EpicKitchensDataset(Dataset):
    """EPIC-KITCHENS dataset for verb classification."""

    def __init__(
        self,
        frames_dir: Path,
        manifest_path: Path,
        class_info_path: Path,
        split: str = "train",
        transform=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.transform = transform

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load class info
        with open(class_info_path) as f:
            self.class_info = json.load(f)

        self.label_map = {int(k): v for k, v in self.class_info["label_map"].items()}
        self.num_classes = self.class_info["num_classes"]

        # Filter to samples with valid labels (top-K verbs only)
        self.samples = []
        for item in manifest.get(split, []):
            verb_class = item["verb_class"]
            if verb_class in self.label_map:
                self.samples.append({
                    "path": self.frames_dir / item["path"],
                    "label": self.label_map[verb_class],
                    "verb": item["verb"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen encoder."""

    def __init__(self, encoder, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            # forward_features returns CLS token directly (shape: [B, embed_dim])
            cls_token = self.encoder.forward_features(x)
        return self.head(cls_token)


def load_mae_encoder(checkpoint_path: Path, model_name: str = "vit_base_patch16"):
    """Load encoder from MAE checkpoint."""
    # Create ViT model (encoder only, no decoder)
    model = models_vit.__dict__[model_name](
        num_classes=0,  # No classification head
        global_pool=False,
    )

    # Load checkpoint (weights_only=False needed for checkpoints with args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Filter to encoder weights only (remove decoder, mask_token, etc.)
    encoder_state_dict = {}
    for k, v in state_dict.items():
        # Handle compiled model prefix
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]

        # Skip decoder and other non-encoder weights
        if k.startswith("decoder") or k.startswith("mask_token"):
            continue

        encoder_state_dict[k] = v

    # Load weights
    msg = model.load_state_dict(encoder_state_dict, strict=False)
    print(f"Loaded MAE encoder: {msg}")

    return model


def load_imagenet_encoder(model_name: str = "vit_base_patch16"):
    """Load ImageNet pretrained ViT encoder."""
    import timm

    # Create model without classification head
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

    # Wrap to match our interface (return CLS token from forward_features)
    class ImageNetEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward_features(self, x):
            # timm's forward_features returns [B, num_patches+1, embed_dim]
            # Extract CLS token (first token)
            features = self.model.forward_features(x)
            return features[:, 0]  # [B, embed_dim]

    return ImageNetEncoder(model)


def load_random_encoder(model_name: str = "vit_base_patch16"):
    """Load randomly initialized ViT encoder."""
    model = models_vit.__dict__[model_name](
        num_classes=0,
        global_pool=False,
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.encoder.eval()  # Keep encoder frozen

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def run_probe(
    encoder,
    encoder_name: str,
    embed_dim: int,
    train_loader,
    val_loader,
    num_classes: int,
    device,
    output_dir: Path,
    epochs: int = 100,
    lr: float = 0.001,
):
    """Run linear probing for a single encoder."""
    print(f"\n{'='*60}")
    print(f"Probing: {encoder_name}")
    print(f"{'='*60}")

    # Create classifier
    encoder = encoder.to(device)
    model = LinearClassifier(encoder, embed_dim, num_classes).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Tensorboard
    log_dir = output_dir / encoder_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Training loop
    best_val_acc = 0
    results = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.head.state_dict(), output_dir / encoder_name / "best_head.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}% (best={best_val_acc:.2f}%)")

    writer.close()

    # Final evaluation with best head
    model.head.load_state_dict(torch.load(output_dir / encoder_name / "best_head.pth", weights_only=True))
    final_loss, final_acc, preds, labels = evaluate(model, val_loader, criterion, device)

    # Save results
    final_results = {
        "encoder": encoder_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_acc,
        "final_val_loss": final_loss,
        "epochs": epochs,
        "lr": lr,
        "history": results,
    }

    with open(output_dir / encoder_name / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Final: val_acc={final_acc:.2f}% (best={best_val_acc:.2f}%)")

    return best_val_acc, final_results


def main():
    parser = argparse.ArgumentParser(description="Linear probing evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to MAE checkpoint")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to epic_kitchens directory")
    parser.add_argument("--output_dir", type=Path, default="./probe_results",
                        help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--model", type=str, default="vit_base_patch16",
                        help="Model architecture")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--skip_imagenet", action="store_true",
                        help="Skip ImageNet pretrained baseline")
    parser.add_argument("--skip_random", action="store_true",
                        help="Skip random initialization baseline")
    parser.add_argument("--skip_mae", action="store_true",
                        help="Skip MAE pretrained evaluation")
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    cudnn.benchmark = True

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = args.checkpoint.stem
    exp_name = f"{timestamp}_{checkpoint_name}"
    output_dir = args.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")

    # Save experiment config
    config = vars(args).copy()
    config["checkpoint"] = str(config["checkpoint"])
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(output_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Paths
    frames_dir = args.data_dir / "EPIC-KITCHENS" / "frames"
    manifest_path = frames_dir / "manifest.json"
    class_info_path = args.data_dir / "class_info.json"

    # Datasets
    print("Loading datasets...")
    train_dataset = EpicKitchensDataset(
        frames_dir, manifest_path, class_info_path,
        split="train", transform=transform_train
    )
    val_dataset = EpicKitchensDataset(
        frames_dir, manifest_path, class_info_path,
        split="val", transform=transform_val
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Get embed dim based on model
    embed_dims = {
        "vit_small_patch16": 384,
        "vit_base_patch16": 768,
        "vit_large_patch16": 1024,
        "vit_huge_patch14": 1280,
    }
    embed_dim = embed_dims.get(args.model, 768)

    # Run probing experiments
    all_results = {}

    # 1. MAE pretrained
    if not args.skip_mae:
        print("\nLoading MAE encoder...")
        mae_encoder = load_mae_encoder(args.checkpoint, args.model)
        mae_acc, mae_results = run_probe(
            mae_encoder, "mae_pretrained", embed_dim,
            train_loader, val_loader, train_dataset.num_classes,
            device, output_dir, args.epochs, args.lr
        )
        all_results["mae_pretrained"] = mae_results

    # 2. Random initialization baseline
    if not args.skip_random:
        print("\nLoading random encoder...")
        random_encoder = load_random_encoder(args.model)
        random_acc, random_results = run_probe(
            random_encoder, "random_init", embed_dim,
            train_loader, val_loader, train_dataset.num_classes,
            device, output_dir, args.epochs, args.lr
        )
        all_results["random_init"] = random_results

    # 3. ImageNet pretrained baseline
    if not args.skip_imagenet:
        print("\nLoading ImageNet encoder...")
        try:
            imagenet_encoder = load_imagenet_encoder(args.model)
            imagenet_acc, imagenet_results = run_probe(
                imagenet_encoder, "imagenet_pretrained", embed_dim,
                train_loader, val_loader, train_dataset.num_classes,
                device, output_dir, args.epochs, args.lr
            )
            all_results["imagenet_pretrained"] = imagenet_results
        except Exception as e:
            print(f"Failed to load ImageNet pretrained: {e}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, results in all_results.items():
        print(f"{name:25s}: {results['best_val_acc']:.2f}%")

    # Save summary
    summary = {
        "experiment": exp_name,
        "checkpoint": str(args.checkpoint),
        "results": {k: v["best_val_acc"] for k, v in all_results.items()},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
