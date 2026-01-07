#!/usr/bin/env python3
"""
Extract videos from Egocentric-10K tar files and count total frames.

Features:
- Parallel extraction of tar files
- Frame counting via metadata JSON (fast) or ffprobe (accurate)
- Progress tracking
- Resume support (skips already extracted files)
"""

import argparse
import asyncio
import json
import os
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    path: Path
    frames: int
    duration: float
    fps: float


def extract_tar(tar_path: Path, output_dir: Path, skip_existing: bool = True) -> list[Path]:
    """Extract a single tar file and return list of extracted video paths."""
    extracted_videos = []

    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()

            for member in members:
                if not member.name.endswith('.mp4') and not member.name.endswith('.json'):
                    continue

                # Determine output path - flatten structure or preserve it
                out_path = output_dir / member.name

                if skip_existing and out_path.exists():
                    if member.name.endswith('.mp4'):
                        extracted_videos.append(out_path)
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract file
                with tar.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(out_path, 'wb') as dst:
                        dst.write(src.read())

                if member.name.endswith('.mp4'):
                    extracted_videos.append(out_path)

    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")

    return extracted_videos


def count_frames_ffprobe(video_path: Path) -> Optional[int]:
    """Count frames using ffprobe (accurate but slower)."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
    return None


def count_frames_metadata(json_path: Path) -> Optional[int]:
    """Count frames from metadata JSON (fast, uses fps * duration)."""
    try:
        with open(json_path) as f:
            meta = json.load(f)
        fps = meta.get('fps', 30.0)
        duration = meta.get('duration_sec', 0)
        return int(fps * duration)
    except Exception as e:
        logger.warning(f"Failed to read metadata {json_path}: {e}")
    return None


def get_video_info(video_path: Path, use_ffprobe: bool = False) -> Optional[VideoInfo]:
    """Get video info including frame count."""
    json_path = video_path.with_suffix('.json')

    # Try metadata first (fast)
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
            fps = meta.get('fps', 30.0)
            duration = meta.get('duration_sec', 0)
            frames = int(fps * duration)

            # Optionally verify with ffprobe
            if use_ffprobe:
                actual_frames = count_frames_ffprobe(video_path)
                if actual_frames:
                    frames = actual_frames

            return VideoInfo(
                path=video_path,
                frames=frames,
                duration=duration,
                fps=fps
            )
        except Exception as e:
            logger.warning(f"Metadata read failed for {video_path}: {e}")

    # Fallback to ffprobe
    if use_ffprobe:
        frames = count_frames_ffprobe(video_path)
        if frames:
            return VideoInfo(
                path=video_path,
                frames=frames,
                duration=0,
                fps=0
            )

    return None


def process_tar_file(args: tuple) -> tuple[list[Path], int]:
    """Process a single tar: extract and count frames. Returns (videos, frame_count)."""
    tar_path, output_dir, skip_existing, use_ffprobe = args

    videos = extract_tar(tar_path, output_dir, skip_existing)
    total_frames = 0

    for video in videos:
        info = get_video_info(video, use_ffprobe)
        if info:
            total_frames += info.frames

    return videos, total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract videos from Egocentric-10K tars and count frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all tars and count frames (using metadata)
  python extract_and_count.py -i ./data -o ./videos

  # Use ffprobe for accurate frame count (slower)
  python extract_and_count.py -i ./data -o ./videos --ffprobe

  # Just count frames without extracting (reads from tar)
  python extract_and_count.py -i ./data --count-only

  # Control parallelism
  python extract_and_count.py -i ./data -o ./videos --workers 8
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input directory containing tar files"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory for extracted videos (default: extract next to tars)"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Number of parallel workers (default: {os.cpu_count() or 4})"
    )

    parser.add_argument(
        "--ffprobe",
        action="store_true",
        help="Use ffprobe for accurate frame counting (slower)"
    )

    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count frames from metadata without extracting"
    )

    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract files even if they exist"
    )

    args = parser.parse_args()

    # Find all tar files
    tar_files = list(args.input.rglob("*.tar"))
    if not tar_files:
        logger.error(f"No tar files found in {args.input}")
        return

    logger.info(f"Found {len(tar_files)} tar files")

    if args.count_only:
        # Fast path: just read metadata from tars without extracting
        total_frames = 0
        total_videos = 0
        total_duration = 0.0

        with tqdm(tar_files, desc="Counting frames", unit="tar") as pbar:
            for tar_path in pbar:
                try:
                    with tarfile.open(tar_path, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('.json'):
                                f = tar.extractfile(member)
                                if f:
                                    meta = json.load(f)
                                    fps = meta.get('fps', 30.0)
                                    duration = meta.get('duration_sec', 0)
                                    frames = int(fps * duration)
                                    total_frames += frames
                                    total_duration += duration
                                    total_videos += 1
                except Exception as e:
                    logger.warning(f"Failed to read {tar_path}: {e}")

                pbar.set_postfix(videos=total_videos, frames=f"{total_frames:,}")

        print(f"\n{'='*50}")
        print(f"Total videos:   {total_videos:,}")
        print(f"Total frames:   {total_frames:,}")
        print(f"Total duration: {total_duration/3600:.1f} hours")
        print(f"{'='*50}")
        return

    # Extraction mode
    output_dir = args.output or args.input / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting to: {output_dir}")
    logger.info(f"Using {args.workers} workers")

    # Prepare work items
    work_items = [
        (tar_path, output_dir, not args.no_skip, args.ffprobe)
        for tar_path in tar_files
    ]

    total_videos = 0
    total_frames = 0
    all_videos = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with tqdm(total=len(work_items), desc="Processing tars", unit="tar") as pbar:
            futures = {executor.submit(process_tar_file, item): item for item in work_items}

            for future in futures:
                try:
                    videos, frames = future.result()
                    all_videos.extend(videos)
                    total_videos += len(videos)
                    total_frames += frames
                except Exception as e:
                    tar_path = futures[future][0]
                    logger.error(f"Failed to process {tar_path}: {e}")

                pbar.update(1)
                pbar.set_postfix(videos=total_videos, frames=f"{total_frames:,}")

    # Final summary
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"Total videos:   {total_videos:,}")
    print(f"Total frames:   {total_frames:,}")
    print(f"Output dir:     {output_dir}")
    print(f"{'='*50}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "total_videos": total_videos,
        "total_frames": total_frames,
        "videos": [str(v) for v in all_videos]
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
