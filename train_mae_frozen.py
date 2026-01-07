#!/usr/bin/env python3
"""
Train MAE with frozen early encoder layers.

Freezes the first N transformer blocks to preserve low-level ImageNet features
while allowing later blocks and decoder to adapt to domain-specific data.

This approach helps prevent catastrophic forgetting when doing continued
pretraining on a new domain.

Usage:
    python train_mae_frozen.py --data_dir ./frames --output_dir ./mae_output_frozen \
        --pretrained_encoder imagenet --freeze_blocks 6 --epochs 50
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Add MAE to path
sys.path.insert(0, str(Path(__file__).parent / "mae"))

import models_mae
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
from util.lr_sched import adjust_learning_rate

import timm


def load_pretrained_encoder(mae_model, source: str, model_name: str = "mae_vit_base_patch16"):
    """Load pretrained weights into MAE encoder (not decoder)."""
    if source == "imagenet":
        # Map MAE model name to timm model name
        timm_model_map = {
            "mae_vit_small_patch16": "vit_small_patch16_224",
            "mae_vit_base_patch16": "vit_base_patch16_224",
            "mae_vit_large_patch16": "vit_large_patch16_224",
        }
        timm_name = timm_model_map.get(model_name, "vit_base_patch16_224")
        print(f"Loading timm model: {timm_name}")
        pretrained = timm.create_model(timm_name, pretrained=True)
        pretrained_dict = pretrained.state_dict()
    else:
        # Load from checkpoint file
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        pretrained_dict = checkpoint.get("model", checkpoint)

    # Map pretrained weights to MAE encoder
    mae_dict = mae_model.state_dict()
    matched = 0
    skipped = 0
    for k, v in pretrained_dict.items():
        # Skip classification head only (head.weight, head.bias)
        # Don't skip MLP layers (fc1, fc2) which are part of the encoder
        if k.startswith("head."):
            skipped += 1
            continue
        # Skip decoder weights if loading from MAE checkpoint
        if "decoder" in k or "mask_token" in k:
            skipped += 1
            continue
        # Match encoder weights
        if k in mae_dict and mae_dict[k].shape == v.shape:
            mae_dict[k] = v
            matched += 1

    mae_model.load_state_dict(mae_dict)
    print(f"Loaded {matched} pretrained encoder weights (skipped {skipped})")


def freeze_early_layers(model, num_blocks: int, freeze_patch_embed: bool = True):
    """Freeze the first N transformer blocks and optionally patch embedding.

    Args:
        model: MAE model (may be compiled)
        num_blocks: Number of encoder blocks to freeze (0-12 for ViT-Base)
        freeze_patch_embed: Whether to also freeze patch embedding layer

    Returns:
        Number of frozen parameters
    """
    # Handle compiled model
    if hasattr(model, '_orig_mod'):
        base_model = model._orig_mod
    else:
        base_model = model

    frozen_params = 0
    total_params = 0

    for name, param in base_model.named_parameters():
        total_params += param.numel()
        should_freeze = False

        # Freeze patch embedding
        if freeze_patch_embed and name.startswith("patch_embed."):
            should_freeze = True

        # Freeze cls_token and pos_embed (these are typically fixed anyway)
        if name in ["cls_token", "pos_embed"]:
            should_freeze = True

        # Freeze early encoder blocks
        if name.startswith("blocks."):
            # Extract block number: blocks.0.attn.qkv.weight -> 0
            block_num = int(name.split(".")[1])
            if block_num < num_blocks:
                should_freeze = True

        # Freeze encoder norm if freezing all blocks
        # (but we typically want to keep this trainable)

        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()

    # Print summary
    trainable_params = total_params - frozen_params
    print(f"\nLayer freezing summary:")
    print(f"  Frozen blocks: 0-{num_blocks-1} (of 12)")
    print(f"  Patch embed frozen: {freeze_patch_embed}")
    print(f"  Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    return frozen_params


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Add weight decay to parameters, skipping bias, norm layers, and frozen params."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class ImageFolderDataset(Dataset):
    """Dataset that loads frames from a directory of images."""

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths: List[Path] = []
        self._last_scan_time = 0
        self._scan_directory()

    def _scan_directory(self):
        """Scan directory for image files."""
        if not self.data_dir.exists():
            return

        new_paths = []
        for ext in self.SUPPORTED_EXTENSIONS:
            new_paths.extend(self.data_dir.glob(f"*{ext}"))
            new_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))

        self.image_paths = sorted(new_paths)
        self._last_scan_time = time.time()

    def reload_if_changed(self, rescan_interval: int = 60) -> bool:
        """Rescan directory for new images. Returns True if count changed."""
        if time.time() - self._last_scan_time < rescan_interval:
            return False

        old_count = len(self.image_paths)
        self._scan_directory()
        new_count = len(self.image_paths)

        if new_count != old_count:
            print(f"Directory rescanned: {old_count} -> {new_count} images")
            return True
        return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training with frozen early layers', add_help=False)

    # Data
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Path to directory containing extracted frames')

    # Training
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs (default: 50 for fine-tuning)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Gradient accumulation steps')

    # Model
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str,
                        choices=['mae_vit_small_patch16', 'mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'])
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')

    # Layer freezing
    parser.add_argument('--freeze_blocks', type=int, default=6,
                        help='Number of encoder blocks to freeze (default: 6 of 12)')
    parser.add_argument('--freeze_patch_embed', action='store_true', default=True,
                        help='Freeze patch embedding layer (default: True)')

    # Optimizer - lower LR for fine-tuning
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-4,
                        help='base lr: lower than from-scratch training (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (default: 5 for fine-tuning)')

    # Output
    parser.add_argument('--output_dir', default='./mae_output_frozen', type=Path)
    parser.add_argument('--log_dir', default=None, type=Path)
    parser.add_argument('--save_every', default=10, type=int,
                        help='Save checkpoint every N epochs')

    # System
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--compile', action='store_true', default=True,
                        help='Use torch.compile for faster training')

    # Live reload
    parser.add_argument('--reload_interval', default=60, type=int,
                        help='Check for new frames every N seconds (0 to disable)')

    # Pretrained initialization (required for this script)
    parser.add_argument('--pretrained_encoder', type=str, default='imagenet',
                        help='Load pretrained encoder (default: "imagenet")')

    return parser


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, 50, header)):
        # Adjust LR per iteration
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and (data_iter_step + 1) % 100 == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(args, epoch, model, optimizer, loss_scaler):
    # Handle compiled model
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch,
        'args': args,
        'frozen_blocks': args.freeze_blocks,
    }
    path = args.output_dir / f'checkpoint-{epoch:04d}.pth'
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

    # Also save as latest
    latest_path = args.output_dir / 'checkpoint-latest.pth'
    torch.save(checkpoint, latest_path)


def load_checkpoint(args, model, optimizer, loss_scaler):
    if args.resume:
        checkpoint_path = Path(args.resume)
    else:
        checkpoint_path = args.output_dir / 'checkpoint-latest.pth'

    if not checkpoint_path.exists():
        return 0

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle compiled model
    if hasattr(model, '_orig_mod'):
        model._orig_mod.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_scaler.load_state_dict(checkpoint['scaler'])

    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def main(args):
    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Args: {args}")

    device = torch.device(args.device)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Create output dirs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = ImageFolderDataset(
        args.data_dir,
        transform=transform_train,
    )
    print(f"Dataset: {len(dataset)} images")

    if len(dataset) == 0:
        print("No images in directory yet. Waiting for data...")
        while len(dataset) == 0:
            time.sleep(10)
            dataset.reload_if_changed(rescan_interval=0)
        print(f"Dataset now has {len(dataset)} images")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # Load pretrained encoder
    if args.pretrained_encoder:
        print(f"Loading pretrained encoder from {args.pretrained_encoder}")
        load_pretrained_encoder(model, args.pretrained_encoder, args.model)

    # Freeze early layers BEFORE moving to device
    freeze_early_layers(model, args.freeze_blocks, args.freeze_patch_embed)

    model.to(device)
    print(f"Model: {args.model}")

    # Compile model for faster training (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Optimizer - only trainable params
    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Base LR: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual LR: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Resume
    start_epoch = load_checkpoint(args, model, optimizer, loss_scaler)

    # Tensorboard
    log_writer = SummaryWriter(log_dir=str(args.log_dir))

    # Save config
    config = {
        'model': args.model,
        'freeze_blocks': args.freeze_blocks,
        'freeze_patch_embed': args.freeze_patch_embed,
        'pretrained_encoder': args.pretrained_encoder,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'blr': args.blr,
        'warmup_epochs': args.warmup_epochs,
    }
    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch} for {args.epochs} epochs")
    start_time = time.time()
    last_reload_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Check for new frames periodically
        if args.reload_interval > 0 and time.time() - last_reload_time > args.reload_interval:
            if dataset.reload_if_changed(rescan_interval=0):
                data_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=True,
                    prefetch_factor=4,
                    persistent_workers=True if args.num_workers > 0 else False,
                )
            last_reload_time = time.time()

        train_stats = train_one_epoch(
            model, data_loader, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args, epoch, model, optimizer, loss_scaler)

        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        with open(args.output_dir / "log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
