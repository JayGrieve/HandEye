#!/usr/bin/env python3
"""
Linear probing evaluation for DINOv3 on EPIC-KITCHENS.

Usage:
    HF_TOKEN=your_token python probe_dinov3.py --data_dir ./epic_kitchens --model vitb16
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel


# DINOv3 model configurations
DINOV3_MODELS = {
    "vits16": {
        "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "embed_dim": 384,
    },
    "vitb16": {
        "hf_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "embed_dim": 768,
    },
    "vitl16": {
        "hf_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "embed_dim": 1024,
    },
}


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

        with open(manifest_path) as f:
            manifest = json.load(f)

        with open(class_info_path) as f:
            self.class_info = json.load(f)

        self.label_map = {int(k): v for k, v in self.class_info["label_map"].items()}
        self.num_classes = self.class_info["num_classes"]

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


class DINOv3Encoder(nn.Module):
    """Wrapper for DINOv3 to provide consistent interface."""

    def __init__(self, hf_model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)

    def forward_features(self, x):
        outputs = self.model(x)
        return outputs.pooler_output  # [B, embed_dim]


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen encoder."""

    def __init__(self, encoder, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder.forward_features(x)
        return self.head(features)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    model.encoder.eval()

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

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="DINOv3 linear probing")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to epic_kitchens directory")
    parser.add_argument("--output_dir", type=Path, default="./probe_results",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="vitb16",
                        choices=list(DINOV3_MODELS.keys()),
                        help="DINOv3 model size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    args = parser.parse_args()

    device = torch.device(args.device)
    cudnn.benchmark = True

    model_config = DINOV3_MODELS[args.model]
    embed_dim = model_config["embed_dim"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_dinov3_{args.model}"
    output_dir = args.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output: {output_dir}")
    print(f"Model: DINOv3 {args.model} (embed_dim={embed_dim})")

    # Save config
    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(output_dir)
    config["hf_model"] = model_config["hf_name"]
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

    # Load DINOv3 encoder
    print(f"\nLoading DINOv3 {args.model}...")
    encoder = DINOv3Encoder(model_config["hf_name"])
    encoder = encoder.to(device)

    # Create classifier
    model = LinearClassifier(encoder, embed_dim, train_dataset.num_classes).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    results_history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "head_state_dict": model.head.state_dict(),
                "best_val_acc": best_val_acc,
            }, output_dir / "best_model.pth")

        results_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

    # Save final results
    final_results = {
        "model": f"dinov3_{args.model}",
        "hf_model": model_config["hf_name"],
        "embed_dim": embed_dim,
        "best_val_acc": best_val_acc,
        "final_val_acc": results_history[-1]["val_acc"],
        "epochs": args.epochs,
        "history": results_history,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - DINOv3 {args.model}")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
