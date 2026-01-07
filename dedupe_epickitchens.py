#!/usr/bin/env python3
"""
Deduplicate EPIC-KITCHENS frames using DINOv3 CLS feature similarity.

Reads frames from extracted RGB frame directories, deduplicates using
DINO features, and saves unique frames to output directory.

Features:
- Works with pre-extracted EPIC-KITCHENS RGB frames (jpg files)
- Incremental saving (writes progress frequently)
- Resume support (skips already processed videos)
- Directly copies selected frames to output directory

Usage:
    python dedupe_epickitchens.py --input ./epic_kitchens/EPIC-KITCHENS --output ./frames_epickitchens
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, List
import logging

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DINOv3FeatureExtractor:
    """Extract CLS features from images using DINOv3."""

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoModel, AutoImageProcessor

        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading DINOv3 model from {model_path} on {self.device}")

        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_path)

        # Get feature dimension from config
        self.feature_dim = self.model.config.hidden_size
        logger.info(f"Feature dimension: {self.feature_dim}")

    @torch.no_grad()
    def extract_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract CLS features from a batch of PIL images.

        Returns:
            Tensor of shape (batch_size, feature_dim) with L2-normalized features
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Get CLS token (first token)
        cls_features = outputs.last_hidden_state[:, 0, :]

        # L2 normalize for cosine similarity
        cls_features = F.normalize(cls_features, p=2, dim=-1)

        return cls_features


def find_video_dirs(input_path: Path) -> List[Path]:
    """Find all video frame directories (e.g., P01/rgb_frames/P01_01/)."""
    video_dirs = []

    # Look for participant directories
    for participant_dir in sorted(input_path.glob("P*")):
        rgb_frames_dir = participant_dir / "rgb_frames"
        if rgb_frames_dir.exists():
            # Each subdirectory is a video
            for video_dir in sorted(rgb_frames_dir.iterdir()):
                if video_dir.is_dir():
                    # Check if it has jpg files
                    if list(video_dir.glob("*.jpg"))[:1]:
                        video_dirs.append(video_dir)

    return video_dirs


def read_frames_from_dir(
    video_dir: Path,
    frame_step: int = 5,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[Path, Image.Image]]:
    """Read frames from a directory at regular intervals.

    Yields:
        Tuples of (frame_path, PIL Image)
    """
    # Get sorted list of frame files
    frame_files = sorted(video_dir.glob("frame_*.jpg"))

    frames_yielded = 0
    for i, frame_path in enumerate(frame_files):
        if i % frame_step == 0:
            try:
                image = Image.open(frame_path).convert('RGB')
                yield frame_path, image
                frames_yielded += 1

                if max_frames and frames_yielded >= max_frames:
                    break
            except Exception as e:
                logger.warning(f"Could not load {frame_path}: {e}")
                continue


def process_video_dir(
    video_dir: Path,
    extractor: DINOv3FeatureExtractor,
    output_dir: Path,
    frame_step: int = 5,
    similarity_threshold: float = 0.90,
    batch_size: int = 32,
    show_progress: bool = True,
) -> dict:
    """Process a single video directory: extract frames, dedupe, copy selected frames.

    Returns:
        Dict with statistics and list of selected frame paths
    """
    kept_features = []  # List of feature tensors for kept frames
    kept_frames = []  # List of frame paths for kept frames
    discarded_count = 0
    total_frames = 0

    # Get frame count for progress bar
    frame_files = list(video_dir.glob("frame_*.jpg"))
    frame_count = len(frame_files)
    sampled_count = (frame_count + frame_step - 1) // frame_step

    # Create progress bar for this video
    pbar = None
    if show_progress and sampled_count > 0:
        pbar = tqdm(
            total=sampled_count,
            desc=f"  {video_dir.name[:30]}",
            unit="frame",
            leave=False
        )

    # Collect frames in batches for efficient processing
    frame_buffer = []
    path_buffer = []

    for frame_path, image in read_frames_from_dir(video_dir, frame_step):
        frame_buffer.append(image)
        path_buffer.append(frame_path)
        total_frames += 1

        # Process batch when full
        if len(frame_buffer) >= batch_size:
            kept, discarded = _process_batch(
                frame_buffer, path_buffer, extractor, kept_features,
                similarity_threshold, kept_frames
            )
            discarded_count += discarded
            frame_buffer = []
            path_buffer = []

            if pbar:
                pbar.update(batch_size)

    # Process remaining frames
    if frame_buffer:
        kept, discarded = _process_batch(
            frame_buffer, path_buffer, extractor, kept_features,
            similarity_threshold, kept_frames
        )
        discarded_count += discarded
        if pbar:
            pbar.update(len(frame_buffer))

    if pbar:
        pbar.close()

    # Copy selected frames to output directory
    copied_paths = []
    for frame_path in kept_frames:
        # Create unique filename: P01_01_frame_0000000001.jpg
        video_name = video_dir.name
        frame_name = frame_path.name
        out_name = f"{video_name}_{frame_name}"
        out_path = output_dir / out_name

        if not out_path.exists():
            shutil.copy(frame_path, out_path)
        copied_paths.append(str(out_path))

    return {
        "video_dir": str(video_dir),
        "total_frames": frame_count,
        "total_sampled": total_frames,
        "kept": len(kept_frames),
        "discarded": discarded_count,
        "dedup_ratio": discarded_count / total_frames if total_frames > 0 else 0,
        "copied_frames": copied_paths
    }


def _process_batch(
    images: list[Image.Image],
    frame_paths: list[Path],
    extractor: DINOv3FeatureExtractor,
    kept_features: list[torch.Tensor],
    threshold: float,
    kept_frames: list[Path],
) -> tuple[int, int]:
    """Process a batch of frames, return (kept_count, discarded_count)."""
    features = extractor.extract_features(images)
    device = features.device

    kept = 0
    discarded = 0

    for i, (feat, frame_path) in enumerate(zip(features, frame_paths)):
        is_duplicate = False

        if kept_features:
            # Stack all kept features and compute similarities
            kept_stack = torch.stack(kept_features).to(device)
            similarities = torch.mv(kept_stack, feat)
            max_sim = similarities.max().item()

            if max_sim >= threshold:
                is_duplicate = True

        if is_duplicate:
            discarded += 1
        else:
            kept += 1
            kept_features.append(feat.cpu())
            kept_frames.append(frame_path)

    return kept, discarded


def load_progress(progress_path: Path) -> tuple[dict, set]:
    """Load existing progress from progress file.

    Returns:
        Tuple of (manifest dict, set of processed video dir names)
    """
    if not progress_path.exists():
        return None, set()

    try:
        with open(progress_path) as f:
            manifest = json.load(f)

        processed = set()
        for result in manifest.get("per_video", []):
            processed.add(result["video_dir"])

        logger.info(f"Resuming: {len(processed)} videos already processed")
        return manifest, processed
    except Exception as e:
        logger.warning(f"Could not load progress: {e}")
        return None, set()


def save_progress(
    progress_path: Path,
    config: dict,
    all_results: list[dict],
    total_kept: int,
    total_discarded: int
):
    """Save current progress to progress file."""
    total_sampled = total_kept + total_discarded

    manifest = {
        "config": config,
        "summary": {
            "videos_processed": len(all_results),
            "total_sampled": total_sampled,
            "total_kept": total_kept,
            "total_discarded": total_discarded,
            "dedup_ratio": total_discarded / total_sampled if total_sampled > 0 else 0
        },
        "per_video": all_results
    }

    # Write to temp file then rename for atomic write
    temp_path = progress_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    temp_path.rename(progress_path)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate EPIC-KITCHENS frames using DINOv3 feature similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process EPIC-KITCHENS frames with default settings (optimized for ~500k output)
  python dedupe_epickitchens.py --input ./epic_kitchens/EPIC-KITCHENS --output ./frames_epickitchens

  # More aggressive dedup (fewer frames)
  python dedupe_epickitchens.py --input ./epic_kitchens/EPIC-KITCHENS --output ./frames_epickitchens --threshold 0.95

  # Less aggressive dedup (more frames)
  python dedupe_epickitchens.py --input ./epic_kitchens/EPIC-KITCHENS --output ./frames_epickitchens --threshold 0.85 --frame-step 3
        """
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input EPIC-KITCHENS directory containing P01/, P02/, etc."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for deduplicated frames"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "dinov3-small",
        help="Path to DINOv3 model directory"
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Sample every N frames (default: 5, lower = more frames)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold for deduplication (default: 0.90, lower = more frames)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save progress every N videos (default: 10)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Progress file in output directory
    progress_path = args.output / "progress.json"

    config = {
        "frame_step": args.frame_step,
        "threshold": args.threshold,
        "model": str(args.model),
        "input": str(args.input),
    }

    # Load existing progress
    existing_manifest, processed_keys = load_progress(progress_path)

    # Load model
    extractor = DINOv3FeatureExtractor(str(args.model), args.device)

    # Find video directories
    video_dirs = find_video_dirs(args.input)
    logger.info(f"Found {len(video_dirs)} video directories")

    # Estimate total frames
    total_raw_frames = sum(len(list(vd.glob("frame_*.jpg"))) for vd in video_dirs)
    estimated_sampled = total_raw_frames // args.frame_step
    logger.info(f"Total raw frames: {total_raw_frames:,}")
    logger.info(f"Estimated frames to sample: {estimated_sampled:,}")

    # Initialize from existing progress
    if existing_manifest:
        all_results = existing_manifest.get("per_video", [])
        total_kept = existing_manifest["summary"]["total_kept"]
        total_discarded = existing_manifest["summary"]["total_discarded"]
    else:
        all_results = []
        total_kept = 0
        total_discarded = 0

    videos_since_save = 0

    def get_stats_str():
        total = total_kept + total_discarded
        if total > 0:
            keep_pct = total_kept / total * 100
            return f"kept={total_kept:,} ({keep_pct:.1f}%)"
        return "kept=0"

    # Process video directories
    pbar = tqdm(video_dirs, desc="Processing videos", position=0)
    for video_dir in pbar:
        key = str(video_dir)
        if key in processed_keys:
            continue  # Skip already processed

        result = process_video_dir(
            video_dir,
            extractor,
            args.output,
            args.frame_step,
            args.threshold,
            args.batch_size,
        )
        all_results.append(result)
        total_kept += result["kept"]
        total_discarded += result["discarded"]
        processed_keys.add(key)
        videos_since_save += 1

        # Update progress bar with stats
        pbar.set_postfix_str(get_stats_str())

        # Save progress periodically
        if videos_since_save >= args.save_every:
            save_progress(progress_path, config, all_results, total_kept, total_discarded)
            videos_since_save = 0

    # Final save
    save_progress(progress_path, config, all_results, total_kept, total_discarded)

    # Summary
    total_sampled = total_kept + total_discarded
    print(f"\n{'='*60}")
    print(f"EPIC-KITCHENS Deduplication complete!")
    print(f"Videos processed:  {len(all_results)}")
    print(f"Frames sampled:    {total_sampled:,}")
    print(f"Frames kept:       {total_kept:,}")
    print(f"Frames discarded:  {total_discarded:,}")
    if total_sampled > 0:
        print(f"Dedup ratio:       {total_discarded/total_sampled*100:.1f}%")
    print(f"Output directory:  {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
