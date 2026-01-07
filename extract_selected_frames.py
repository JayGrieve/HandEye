#!/usr/bin/env python3
"""
Extract all selected frames from videos to a directory.

Usage:
    python extract_selected_frames.py --manifest ./selected_frames.json --output ./frames
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict
import hashlib
import multiprocessing as mp
from multiprocessing import Pool, Manager
import cv2
from PIL import Image
from tqdm import tqdm
import shutil


def extract_single_frame(args):
    """Extract a single frame from a video in a tar file."""
    tar_path, video_name, frame_idx, output_path, video_cache_dir = args

    try:
        # Check if already exists
        if os.path.exists(output_path):
            return (True, None)

        # Create cache key for video
        cache_key = hashlib.md5(f"{tar_path}::{video_name}".encode()).hexdigest()[:16]
        video_cache_path = os.path.join(video_cache_dir, f"{cache_key}.mp4")

        # Extract video to cache if needed
        if not os.path.exists(video_cache_path):
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    for member in tar.getmembers():
                        if member.name == video_name or member.name.endswith("/" + video_name) or os.path.basename(member.name) == video_name:
                            video_data = tar.extractfile(member)
                            if video_data:
                                # Write to temp file first, then rename
                                tmp_path = video_cache_path + ".tmp"
                                with open(tmp_path, 'wb') as f:
                                    f.write(video_data.read())
                                os.rename(tmp_path, video_cache_path)
                            break
            except Exception as e:
                return (False, f"Tar error: {e}")

        if not os.path.exists(video_cache_path):
            return (False, "Video not found in tar")

        # Extract frame
        cap = cv2.VideoCapture(video_cache_path)
        if not cap.isOpened():
            return (False, "Cannot open video")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return (False, "Cannot read frame")

        # Save frame
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.save(output_path, quality=95)

        return (True, None)

    except Exception as e:
        return (False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Extract selected frames to a directory")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to selected_frames.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for frames")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--video-cache", type=Path, default=None, help="Directory to cache extracted videos")
    parser.add_argument("--skip-existing", action="store_true", help="Skip frames that already exist")
    args = parser.parse_args()

    if args.workers is None:
        args.workers = min(mp.cpu_count(), 32)

    # Setup video cache
    if args.video_cache is None:
        args.video_cache = Path(tempfile.gettempdir()) / "frame_extract_video_cache"
    args.video_cache.mkdir(parents=True, exist_ok=True)
    print(f"Video cache: {args.video_cache}")

    # Load manifest
    print("Loading manifest...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    frames = manifest.get("selected_frames", [])
    print(f"Found {len(frames)} frames in manifest")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Build work items
    print("Building work items...")
    work_items = []
    skipped = 0
    seen_outputs = set()

    for frame_info in frames:
        tar_path = frame_info.get("tar_path")
        video_name = frame_info.get("video_name")
        frame_idx = frame_info.get("frame_idx", 0)

        # Create unique filename
        video_stem = Path(video_name).stem if video_name else "unknown"
        output_filename = f"{video_stem}_frame{frame_idx:06d}.jpg"
        output_path = args.output / output_filename

        # Handle duplicates in manifest
        if str(output_path) in seen_outputs:
            key = f"{tar_path}::{video_name}::{frame_idx}"
            hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
            output_filename = f"{video_stem}_frame{frame_idx:06d}_{hash_suffix}.jpg"
            output_path = args.output / output_filename

        seen_outputs.add(str(output_path))

        # Skip existing
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue

        work_items.append((
            tar_path,
            video_name,
            frame_idx,
            str(output_path),
            str(args.video_cache)
        ))

    print(f"Processing {len(work_items)} frames")
    print(f"Skipped {skipped} existing frames")
    print(f"Using {args.workers} workers")

    if not work_items:
        print("Nothing to do!")
        return

    # Process with multiprocessing
    successful = 0
    failed = 0

    print("Starting extraction...")

    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(extract_single_frame, work_items, chunksize=10)

        with tqdm(total=len(work_items), desc="Extracting", unit="frame") as pbar:
            for success, error in results:
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=successful, fail=failed)

    print(f"\nDone!")
    print(f"Extracted: {successful:,} frames")
    print(f"Failed: {failed:,} frames")
    print(f"Output: {args.output}")

    # Optionally clean up video cache
    cache_size = sum(f.stat().st_size for f in args.video_cache.iterdir() if f.is_file())
    print(f"Video cache size: {cache_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
