#!/usr/bin/env python3
"""
Download EPIC-KITCHENS-100 subset for action recognition evaluation.

Downloads RGB frames for a few participants and extracts frames with labels
for training a classifier as a downstream evaluation task.

Usage:
    python download_epic_kitchens.py --output ./epic_kitchens --participants 1 2 3
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
import urllib.request
import csv
import json
import shutil
from collections import defaultdict
import random


ANNOTATIONS_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master"
ANNOTATIONS = {
    "train": f"{ANNOTATIONS_URL}/EPIC_100_train.csv",
    "validation": f"{ANNOTATIONS_URL}/EPIC_100_validation.csv",
    "verb_classes": f"{ANNOTATIONS_URL}/EPIC_100_verb_classes.csv",
    "noun_classes": f"{ANNOTATIONS_URL}/EPIC_100_noun_classes.csv",
}


def download_file(url: str, output_path: Path, desc: str = None):
    """Download a file with progress."""
    if output_path.exists():
        print(f"  {desc or output_path.name} already exists, skipping")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url}...")

    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_annotations(output_dir: Path):
    """Download annotation files."""
    print("Downloading annotations...")
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for name, url in ANNOTATIONS.items():
        download_file(url, ann_dir / f"{name}.csv", name)

    return ann_dir


def parse_annotations(ann_dir: Path, participants: list):
    """Parse annotations and filter by participant."""

    # Load verb classes
    verb_classes = {}
    with open(ann_dir / "verb_classes.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verb_classes[int(row['id'])] = row['key']

    # Load noun classes
    noun_classes = {}
    with open(ann_dir / "noun_classes.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noun_classes[int(row['id'])] = row['key']

    # Parse train and validation annotations
    annotations = {"train": [], "validation": []}

    for split in ["train", "validation"]:
        csv_path = ann_dir / f"{split}.csv"
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by participant
                participant_id = int(row['participant_id'].replace('P', ''))
                if participants and participant_id not in participants:
                    continue

                annotations[split].append({
                    "narration_id": row['narration_id'],
                    "participant_id": row['participant_id'],
                    "video_id": row['video_id'],
                    "start_frame": int(row['start_frame']),
                    "stop_frame": int(row['stop_frame']),
                    "verb": row['verb'],
                    "verb_class": int(row['verb_class']),
                    "noun": row['noun'],
                    "noun_class": int(row['noun_class']),
                    "narration": row['narration'],
                })

    return annotations, verb_classes, noun_classes


def clone_download_scripts(output_dir: Path):
    """Clone the official download scripts."""
    scripts_dir = output_dir / "epic-kitchens-download-scripts"

    if scripts_dir.exists():
        print("Download scripts already cloned")
        return scripts_dir

    print("Cloning EPIC-KITCHENS download scripts...")
    subprocess.run([
        "git", "clone",
        "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git",
        str(scripts_dir)
    ], check=True)

    return scripts_dir


def download_rgb_frames(scripts_dir: Path, output_dir: Path, participants: list):
    """Download RGB frames using official scripts."""

    print(f"Downloading RGB frames for participants: {participants}")

    # Build participant string (comma-separated)
    participant_str = ",".join(str(p) for p in participants)

    cmd = [
        sys.executable,
        "epic_downloader.py",
        "--rgb-frames",
        "--output-path", str(output_dir.absolute()),
        "--participants", participant_str,
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"  in directory: {scripts_dir}")
    subprocess.run(cmd, cwd=str(scripts_dir.absolute()))


def extract_tar_files(output_dir: Path):
    """Extract all tar files in the EPIC-KITCHENS directory."""
    import tarfile

    epic_dir = output_dir / "EPIC-KITCHENS"
    tar_files = list(epic_dir.glob("**/rgb_frames/*.tar"))

    if not tar_files:
        print("No tar files found to extract")
        return

    print(f"Found {len(tar_files)} tar files to extract...")

    for tar_path in tar_files:
        extract_dir = tar_path.parent / tar_path.stem

        if extract_dir.exists() and list(extract_dir.glob("*.jpg")):
            print(f"  {tar_path.name} already extracted, skipping")
            continue

        print(f"  Extracting {tar_path.name}...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
        except Exception as e:
            print(f"    Error extracting {tar_path}: {e}")

    print("Extraction complete!")


def extract_action_frames(output_dir: Path, annotations: dict, num_frames_per_action: int = 5):
    """Extract middle frames from each action segment for classification."""

    # First extract any tar files
    extract_tar_files(output_dir)

    frames_dir = output_dir / "EPIC-KITCHENS" / "frames"
    rgb_dir = output_dir / "EPIC-KITCHENS"

    # Find where RGB frames are stored
    possible_paths = [
        rgb_dir,
        output_dir / "rgb_frames",
    ]

    rgb_root = None
    for p in possible_paths:
        if p.exists():
            # Look for participant folders
            if list(p.glob("P*")):
                rgb_root = p
                break

    if rgb_root is None:
        print("RGB frames not found. Run download first.")
        return

    print(f"Found RGB frames at: {rgb_root}")

    # Create output directories
    for split in ["train", "val"]:
        (frames_dir / split).mkdir(parents=True, exist_ok=True)

    # Extract frames
    extracted = {"train": [], "val": []}

    for split, split_name in [("train", "train"), ("validation", "val")]:
        print(f"Extracting {split} frames...")

        for ann in annotations[split]:
            video_id = ann["video_id"]
            participant_id = ann["participant_id"]
            start_frame = ann["start_frame"]
            stop_frame = ann["stop_frame"]

            # Find RGB frames directory for this video
            # Structure: EPIC-KITCHENS/P01/rgb_frames/P01_01/frame_*.jpg
            video_rgb_dir = rgb_root / participant_id / "rgb_frames" / video_id

            if not video_rgb_dir.exists():
                # Try alternative structures
                alt_paths = [
                    rgb_root / participant_id / video_id,
                    rgb_root / participant_id / "rgb_frames" / video_id,
                ]
                video_rgb_dir = None
                for alt in alt_paths:
                    if alt.exists():
                        video_rgb_dir = alt
                        break
                if video_rgb_dir is None:
                    continue

            # Get middle frames from the action segment
            duration = stop_frame - start_frame
            if duration < num_frames_per_action:
                frame_indices = list(range(start_frame, stop_frame + 1))
            else:
                step = duration // num_frames_per_action
                frame_indices = [start_frame + i * step for i in range(num_frames_per_action)]

            for idx, frame_idx in enumerate(frame_indices):
                # EPIC-KITCHENS frame naming: frame_0000000001.jpg
                frame_name = f"frame_{frame_idx:010d}.jpg"
                frame_path = video_rgb_dir / frame_name

                if not frame_path.exists():
                    continue

                # Create output filename with label info
                verb_class = ann["verb_class"]
                noun_class = ann["noun_class"]
                out_name = f"{ann['narration_id']}_{idx}_v{verb_class}_n{noun_class}.jpg"
                out_path = frames_dir / split_name / out_name

                shutil.copy(frame_path, out_path)

                extracted[split_name].append({
                    "path": str(out_path.relative_to(frames_dir)),
                    "verb_class": verb_class,
                    "noun_class": noun_class,
                    "verb": ann["verb"],
                    "noun": ann["noun"],
                    "narration": ann["narration"],
                })

        print(f"  Extracted {len(extracted[split_name])} frames")

    # Save manifest
    manifest_path = frames_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(extracted, f, indent=2)

    print(f"Saved manifest to {manifest_path}")
    return extracted


def create_simple_dataset(output_dir: Path, annotations: dict, verb_classes: dict,
                          noun_classes: dict, top_k_verbs: int = 20):
    """Create a simple image classification dataset using top-K verb classes."""

    print(f"\nCreating simplified dataset with top {top_k_verbs} verb classes...")

    # Count verb occurrences
    verb_counts = defaultdict(int)
    for ann in annotations["train"]:
        verb_counts[ann["verb_class"]] += 1

    # Get top-K verbs
    top_verbs = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)[:top_k_verbs]
    top_verb_ids = {v[0] for v in top_verbs}

    print("Top verb classes:")
    for verb_id, count in top_verbs:
        print(f"  {verb_id}: {verb_classes.get(verb_id, 'unknown')} ({count} samples)")

    # Create label mapping (0 to top_k-1)
    label_map = {verb_id: idx for idx, (verb_id, _) in enumerate(top_verbs)}

    # Save class info
    class_info = {
        "num_classes": top_k_verbs,
        "label_map": {str(k): v for k, v in label_map.items()},
        "class_names": {str(label_map[verb_id]): verb_classes.get(verb_id, 'unknown')
                        for verb_id in top_verb_ids},
    }

    class_info_path = output_dir / "class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(class_info, f, indent=2)

    print(f"Saved class info to {class_info_path}")

    return label_map, top_verb_ids


def main():
    parser = argparse.ArgumentParser(description="Download EPIC-KITCHENS for evaluation")
    parser.add_argument("--output", type=Path, default="./epic_kitchens",
                        help="Output directory")
    parser.add_argument("--participants", type=int, nargs="+", default=[1, 2, 3],
                        help="Participant IDs to download (default: 1 2 3)")
    parser.add_argument("--download-frames", action="store_true",
                        help="Download RGB frames (large, ~50GB per participant)")
    parser.add_argument("--frames-per-action", type=int, default=3,
                        help="Number of frames to extract per action segment")
    parser.add_argument("--top-k-verbs", type=int, default=20,
                        help="Use top-K most common verb classes")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract tars and create dataset (no download)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotations
    ann_dir = download_annotations(args.output)

    # Step 2: Parse annotations
    print("\nParsing annotations...")
    annotations, verb_classes, noun_classes = parse_annotations(ann_dir, args.participants)

    print(f"Train annotations: {len(annotations['train'])}")
    print(f"Validation annotations: {len(annotations['validation'])}")
    print(f"Verb classes: {len(verb_classes)}")
    print(f"Noun classes: {len(noun_classes)}")

    # Step 3: Create simplified dataset info
    label_map, top_verb_ids = create_simple_dataset(
        args.output, annotations, verb_classes, noun_classes, args.top_k_verbs
    )

    # Step 4: Optionally download RGB frames
    if args.download_frames:
        scripts_dir = clone_download_scripts(args.output)
        download_rgb_frames(scripts_dir, args.output, args.participants)

        # Step 5: Extract action frames
        extract_action_frames(args.output, annotations, args.frames_per_action)
    elif args.extract_only:
        # Just extract existing tars and create dataset
        extract_action_frames(args.output, annotations, args.frames_per_action)
    else:
        print("\n" + "="*60)
        print("Annotations downloaded. To download RGB frames, run:")
        print(f"  python {__file__} --output {args.output} --participants {' '.join(map(str, args.participants))} --download-frames")
        print("="*60)
        print("\nAlternatively, download manually from:")
        print("  https://github.com/epic-kitchens/epic-kitchens-download-scripts")
        print("\nOr use Academic Torrents (faster):")
        print("  https://academictorrents.com/details/d08f4591f118a9ab39c1b1f89eab8adc27e34de8")
        print("\nOnce you have tars, run with --extract-only to create dataset:"
              f"\n  python {__file__} --output {args.output} --participants {' '.join(map(str, args.participants))} --extract-only")

    # Save summary
    summary = {
        "participants": args.participants,
        "train_samples": len(annotations["train"]),
        "val_samples": len(annotations["validation"]),
        "verb_classes": len(verb_classes),
        "noun_classes": len(noun_classes),
        "top_k_verbs": args.top_k_verbs,
    }

    with open(args.output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Output saved to {args.output}")


if __name__ == "__main__":
    main()
