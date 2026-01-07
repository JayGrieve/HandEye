#!/usr/bin/env python3
"""
Deduplicate video frames using DINOv3 CLS feature similarity.

Reads every N frames from videos, extracts DINOv3 CLS features,
and outputs a manifest of unique frames (no files are moved or deleted).

Features:
- Incremental saving (writes progress frequently)
- Resume support (skips already processed videos)
- Progress bars for both videos and frames

Usage:
    python dedupe_frames.py -i ./videos -o ./selected_frames.json --threshold 0.95
    python dedupe_frames.py -i ./data -o ./selected_frames.json --from-tar
"""

import argparse
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator
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


@dataclass
class FrameResult:
    video_path: str
    frame_idx: int
    kept: bool
    similarity: float


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

    def extract_single(self, image: Image.Image) -> torch.Tensor:
        """Extract features from a single image."""
        return self.extract_features([image])[0]


def get_video_frame_count(video_path: Path) -> int:
    """Get total frame count of a video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def read_video_frames(
    video_path: Path,
    frame_step: int = 10,
    max_frames: Optional[int] = None,
    pbar: Optional[tqdm] = None
) -> Iterator[tuple[int, Image.Image]]:
    """Read frames from a video file at regular intervals.

    Yields:
        Tuples of (frame_index, PIL Image)
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return

    frame_idx = 0
    frames_yielded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            yield frame_idx, image
            frames_yielded += 1

            if pbar:
                pbar.update(frame_step)

            if max_frames and frames_yielded >= max_frames:
                break

        frame_idx += 1

    cap.release()


def process_video(
    video_path: Path,
    extractor: DINOv3FeatureExtractor,
    frame_step: int = 10,
    similarity_threshold: float = 0.95,
    batch_size: int = 16,
    show_progress: bool = True,
) -> dict:
    """Process a single video: extract frames, dedupe, return selected frame indices.

    Returns:
        Dict with statistics and list of selected frame indices
    """
    kept_features = []  # List of feature tensors for kept frames
    kept_frames = []  # List of (frame_idx) for kept frames
    discarded_count = 0
    total_frames = 0

    # Get frame count for progress bar
    frame_count = get_video_frame_count(video_path)

    # Create progress bar for this video
    pbar = None
    if show_progress and frame_count > 0:
        pbar = tqdm(
            total=frame_count,
            desc=f"  {video_path.name[:30]}",
            unit="frame",
            leave=False
        )

    # Collect frames in batches for efficient processing
    frame_buffer = []
    frame_indices = []

    for frame_idx, image in read_video_frames(video_path, frame_step, pbar=pbar):
        frame_buffer.append(image)
        frame_indices.append(frame_idx)
        total_frames += 1

        # Process batch when full
        if len(frame_buffer) >= batch_size:
            kept, discarded = _process_batch(
                frame_buffer, frame_indices, extractor, kept_features,
                similarity_threshold, kept_frames
            )
            discarded_count += discarded
            frame_buffer = []
            frame_indices = []

    # Process remaining frames
    if frame_buffer:
        kept, discarded = _process_batch(
            frame_buffer, frame_indices, extractor, kept_features,
            similarity_threshold, kept_frames
        )
        discarded_count += discarded

    if pbar:
        pbar.close()

    return {
        "video": str(video_path),
        "total_sampled": total_frames,
        "kept": len(kept_frames),
        "discarded": discarded_count,
        "dedup_ratio": discarded_count / total_frames if total_frames > 0 else 0,
        "selected_frames": kept_frames
    }


def _process_batch(
    images: list[Image.Image],
    frame_indices: list[int],
    extractor: DINOv3FeatureExtractor,
    kept_features: list[torch.Tensor],
    threshold: float,
    kept_frames: list[int],
) -> tuple[int, int]:
    """Process a batch of frames, return (kept_count, discarded_count)."""
    features = extractor.extract_features(images)
    device = features.device

    kept = 0
    discarded = 0

    for i, (feat, frame_idx) in enumerate(zip(features, frame_indices)):
        is_duplicate = False

        if kept_features:
            # Stack all kept features and compute similarities (move to same device)
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
            kept_frames.append(frame_idx)

    return kept, discarded


def find_videos(input_path: Path, from_tar: bool = False) -> list[Path]:
    """Find all video files in input path."""
    if from_tar:
        # Return tar files - we'll extract videos from them
        return sorted(input_path.rglob("*.tar"))
    else:
        # Find mp4 files directly
        videos = list(input_path.rglob("*.mp4"))
        videos.extend(input_path.rglob("*.avi"))
        videos.extend(input_path.rglob("*.mov"))
        return sorted(videos)


def process_tar_video(
    tar_path: Path,
    extractor: DINOv3FeatureExtractor,
    frame_step: int,
    similarity_threshold: float,
    batch_size: int,
) -> list[dict]:
    """Extract and process videos from a tar file."""
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract mp4 files from tar
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.mp4'):
                    tar.extract(member, tmpdir)

        # Process extracted videos
        for video_path in sorted(tmpdir.rglob("*.mp4")):
            result = process_video(
                video_path,
                extractor,
                frame_step,
                similarity_threshold,
                batch_size,
                show_progress=True,
            )
            # Store tar path and video name for reference
            result["tar_path"] = str(tar_path)
            result["video_name"] = video_path.name
            results.append(result)

    return results


def get_video_key(result: dict) -> str:
    """Get a unique key for a video result."""
    if "tar_path" in result:
        return f"{result['tar_path']}::{result['video_name']}"
    return result["video"]


def load_progress(output_path: Path) -> tuple[dict, set]:
    """Load existing progress from output file.

    Returns:
        Tuple of (manifest dict, set of processed video keys)
    """
    if not output_path.exists():
        return None, set()

    try:
        with open(output_path) as f:
            manifest = json.load(f)

        processed = set()
        for result in manifest.get("per_video", []):
            processed.add(get_video_key(result))

        logger.info(f"Resuming from {output_path}: {len(processed)} videos already processed")
        return manifest, processed
    except Exception as e:
        logger.warning(f"Could not load progress from {output_path}: {e}")
        return None, set()


def save_progress(
    output_path: Path,
    config: dict,
    all_results: list[dict],
    total_kept: int,
    total_discarded: int
):
    """Save current progress to output file."""
    # Build selected frames manifest
    selected_frames = []
    for result in all_results:
        video_path = result.get("video")
        tar_path = result.get("tar_path")
        video_name = result.get("video_name", Path(video_path).name if video_path else None)

        for frame_idx in result["selected_frames"]:
            entry = {
                "frame_idx": frame_idx,
            }
            if tar_path:
                entry["tar_path"] = tar_path
                entry["video_name"] = video_name
            else:
                entry["video_path"] = video_path
            selected_frames.append(entry)

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
        "selected_frames": selected_frames,
        "per_video": all_results
    }

    # Write to temp file then rename for atomic write
    temp_path = output_path.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    temp_path.rename(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate video frames using DINOv3 feature similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process extracted videos, output manifest
  python dedupe_frames.py -i ./videos -o ./selected_frames.json

  # Process directly from tar files
  python dedupe_frames.py -i ./data -o ./selected_frames.json --from-tar

  # Adjust threshold (higher = more aggressive dedup)
  python dedupe_frames.py -i ./videos -o ./selected.json --threshold 0.98

  # Sample every 30 frames instead of 10
  python dedupe_frames.py -i ./videos -o ./selected.json --frame-step 30
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input directory containing videos or tar files"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output JSON file with selected frame manifest"
    )

    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "dinov3-small",
        help="Path to DINOv3 model directory"
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Sample every N frames (default: 10)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for deduplication (default: 0.95)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for feature extraction (default: 16)"
    )

    parser.add_argument(
        "--from-tar",
        action="store_true",
        help="Read videos from tar files instead of extracted files"
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
        default=5,
        help="Save progress every N videos (default: 5)"
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "frame_step": args.frame_step,
        "threshold": args.threshold,
        "model": str(args.model),
    }

    # Load existing progress
    existing_manifest, processed_keys = load_progress(args.output)

    # Load model
    extractor = DINOv3FeatureExtractor(str(args.model), args.device)

    # Find videos
    if args.from_tar:
        tar_files = find_videos(args.input, from_tar=True)
        logger.info(f"Found {len(tar_files)} tar files")
    else:
        videos = find_videos(args.input, from_tar=False)
        logger.info(f"Found {len(videos)} videos")

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

    if args.from_tar:
        # Process tar files
        pbar = tqdm(tar_files, desc="Processing tars", position=0)
        for tar_path in pbar:
            # Quick check: see if this tar's videos are likely already processed
            # by checking if any key starting with this tar path exists
            tar_str = str(tar_path)
            tar_already_done = any(k.startswith(tar_str + "::") for k in processed_keys)
            if tar_already_done:
                continue  # Skip this tar entirely

            results = process_tar_video(
                tar_path,
                extractor,
                args.frame_step,
                args.threshold,
                args.batch_size,
            )

            for result in results:
                key = get_video_key(result)
                if key in processed_keys:
                    continue  # Skip already processed

                all_results.append(result)
                total_kept += result["kept"]
                total_discarded += result["discarded"]
                processed_keys.add(key)
                videos_since_save += 1

                # Update progress bar with stats
                pbar.set_postfix_str(get_stats_str())

                # Save progress periodically
                if videos_since_save >= args.save_every:
                    save_progress(args.output, config, all_results, total_kept, total_discarded)
                    videos_since_save = 0
    else:
        # Process videos directly
        pbar = tqdm(videos, desc="Processing videos", position=0)
        for video_path in pbar:
            key = str(video_path)
            if key in processed_keys:
                continue  # Skip already processed

            result = process_video(
                video_path,
                extractor,
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
                save_progress(args.output, config, all_results, total_kept, total_discarded)
                videos_since_save = 0

    # Final save
    save_progress(args.output, config, all_results, total_kept, total_discarded)

    # Summary
    total_sampled = total_kept + total_discarded
    print(f"\n{'='*50}")
    print(f"Deduplication complete!")
    print(f"Videos processed:  {len(all_results)}")
    print(f"Frames sampled:    {total_sampled:,}")
    print(f"Frames kept:       {total_kept:,}")
    print(f"Frames discarded:  {total_discarded:,}")
    if total_sampled > 0:
        print(f"Dedup ratio:       {total_discarded/total_sampled*100:.1f}%")
    print(f"Output file:       {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
