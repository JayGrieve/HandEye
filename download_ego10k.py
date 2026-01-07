#!/usr/bin/env python3
"""
Fast parallel downloader for Hugging Face Egocentric-10K dataset.
https://huggingface.co/datasets/builddotai/Egocentric-10K

Features:
- Parallel downloads with configurable concurrency
- Resume support (skips already downloaded files)
- Progress tracking per file and overall
- Retry logic with exponential backoff
- Optional filtering by factory/worker
"""

import argparse
import asyncio
import aiohttp
import aiofiles
import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tqdm.asyncio import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_ID = "builddotai/Egocentric-10K"
HF_API_BASE = "https://huggingface.co/api/datasets"
HF_RESOLVE_BASE = "https://huggingface.co/datasets"
NUM_FACTORIES = 85


@dataclass
class FileInfo:
    path: str
    size: int
    url: str
    local_path: Path


async def get_tree(
    session: aiohttp.ClientSession,
    path: str = "",
    retries: int = 8,
    base_delay: float = 2.0
) -> list[dict]:
    """Fetch directory tree from HF API with retry logic for rate limits."""
    url = f"{HF_API_BASE}/{DATASET_ID}/tree/main"
    if path:
        url += f"/{path}"

    for attempt in range(retries):
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 429:
                # Rate limited - exponential backoff with jitter
                import random
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.debug(f"Rate limited for {path}, waiting {delay:.1f}s (attempt {attempt + 1}/{retries})")
                await asyncio.sleep(delay)
            else:
                logger.warning(f"Failed to fetch tree for {path}: {resp.status}")
                return []

    logger.warning(f"Failed to fetch tree for {path} after {retries} retries (rate limited)")
    return []


async def discover_files(
    session: aiohttp.ClientSession,
    factories: Optional[list[int]] = None,
    workers: Optional[list[int]] = None,
    semaphore: asyncio.Semaphore = None
) -> list[FileInfo]:
    """Discover all tar files in the dataset."""
    files = []

    # Determine which factories to scan
    factory_range = factories if factories else range(1, NUM_FACTORIES + 1)

    async def scan_worker(factory_id: int, worker_id: int) -> list[FileInfo]:
        """Scan a single worker directory for tar files."""
        if semaphore:
            async with semaphore:
                return await _scan_worker(factory_id, worker_id)
        return await _scan_worker(factory_id, worker_id)

    async def _scan_worker(factory_id: int, worker_id: int) -> list[FileInfo]:
        path = f"factory_{factory_id:03d}/workers/worker_{worker_id:03d}"
        items = await get_tree(session, path)
        worker_files = []
        for item in items:
            if item.get("type") == "file" and item["path"].endswith(".tar"):
                file_path = item["path"]
                url = f"{HF_RESOLVE_BASE}/{DATASET_ID}/resolve/main/{file_path}"
                worker_files.append(FileInfo(
                    path=file_path,
                    size=item.get("size", 0),
                    url=url,
                    local_path=Path(file_path)
                ))
        return worker_files

    async def scan_factory(factory_id: int) -> list[FileInfo]:
        """Scan a factory for all workers."""
        path = f"factory_{factory_id:03d}/workers"
        if semaphore:
            async with semaphore:
                items = await get_tree(session, path)
        else:
            items = await get_tree(session, path)

        factory_files = []
        worker_tasks = []

        for item in items:
            if item.get("type") == "directory" and "worker_" in item["path"]:
                # Extract worker ID from path
                worker_name = item["path"].split("/")[-1]
                worker_id = int(worker_name.replace("worker_", ""))

                # Filter by workers if specified
                if workers and worker_id not in workers:
                    continue

                worker_tasks.append(scan_worker(factory_id, worker_id))

        results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                factory_files.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Error scanning worker: {result}")

        return factory_files

    # Scan all factories in parallel
    logger.info(f"Discovering files in {len(list(factory_range))} factories...")
    tasks = [scan_factory(f) for f in factory_range]

    with tqdm(total=len(tasks), desc="Scanning factories", unit="factory") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            files.extend(result)
            pbar.update(1)

    return files


async def download_file(
    session: aiohttp.ClientSession,
    file_info: FileInfo,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    retries: int = 3,
    hf_token: Optional[str] = None
) -> bool:
    """Download a single file with resume support and retries."""
    local_path = output_dir / file_info.local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is complete
    if local_path.exists():
        existing_size = local_path.stat().st_size
        if existing_size == file_info.size:
            pbar.update(file_info.size)
            return True
        # Partial download - will resume
        start_byte = existing_size
    else:
        start_byte = 0

    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    for attempt in range(retries):
        try:
            async with semaphore:
                async with session.get(file_info.url, headers=headers) as resp:
                    if resp.status == 416:  # Range not satisfiable - file complete
                        pbar.update(file_info.size - start_byte)
                        return True

                    if resp.status not in (200, 206):
                        raise aiohttp.ClientError(f"HTTP {resp.status}")

                    mode = "ab" if start_byte > 0 else "wb"
                    async with aiofiles.open(local_path, mode) as f:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                            await f.write(chunk)
                            pbar.update(len(chunk))

            return True

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait_time = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {file_info.path}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(wait_time)

    logger.error(f"Failed to download {file_info.path} after {retries} attempts")
    return False


async def download_dataset(
    output_dir: Path,
    max_concurrent: int = 8,
    factories: Optional[list[int]] = None,
    workers: Optional[list[int]] = None,
    hf_token: Optional[str] = None,
    dry_run: bool = False
):
    """Main download orchestrator."""

    connector = aiohttp.TCPConnector(limit=max_concurrent * 2, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=3600, connect=30)

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        # Discovery phase - use lower concurrency to avoid rate limits
        discovery_sem = asyncio.Semaphore(3)  # Limit concurrent API requests
        files = await discover_files(session, factories, workers, discovery_sem)

        if not files:
            logger.error("No files found to download!")
            return

        # Calculate totals
        total_size = sum(f.size for f in files)
        total_size_gb = total_size / (1024 ** 3)

        logger.info(f"Found {len(files)} files totaling {total_size_gb:.2f} GB")

        if dry_run:
            logger.info("Dry run - not downloading")
            for f in files[:20]:
                print(f"  {f.path} ({f.size / (1024**2):.1f} MB)")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more files")
            return

        # Check existing files
        existing_size = 0
        for f in files:
            local_path = output_dir / f.local_path
            if local_path.exists():
                existing_size += min(local_path.stat().st_size, f.size)

        remaining_size = total_size - existing_size
        logger.info(f"Already downloaded: {existing_size / (1024**3):.2f} GB")
        logger.info(f"Remaining: {remaining_size / (1024**3):.2f} GB")

        # Download phase
        semaphore = asyncio.Semaphore(max_concurrent)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            # Update progress for existing files
            pbar.update(existing_size)

            tasks = [
                download_file(session, f, output_dir, semaphore, pbar, hf_token=hf_token)
                for f in files
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Report results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful

        logger.info(f"Download complete: {successful}/{len(files)} files successful")
        if failed > 0:
            logger.warning(f"{failed} files failed to download")


def parse_range(range_str: str) -> list[int]:
    """Parse a range string like '1-5,10,15-20' into a list of integers."""
    result = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel downloader for Egocentric-10K dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire dataset (16.4 TB)
  python download_ego10k.py -o ./data

  # Download only factory 1-5 with 16 parallel downloads
  python download_ego10k.py -o ./data --factories 1-5 --parallel 16

  # Download specific workers from factory 1
  python download_ego10k.py -o ./data --factories 1 --workers 1-3,5

  # Dry run to see what would be downloaded
  python download_ego10k.py -o ./data --factories 1 --dry-run

  # Use HF token for gated datasets (set HF_TOKEN env var)
  HF_TOKEN=your_token python download_ego10k.py -o ./data
        """
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./egocentric-10k"),
        help="Output directory (default: ./egocentric-10k)"
    )

    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=8,
        help="Number of parallel downloads (default: 8)"
    )

    parser.add_argument(
        "--factories",
        type=str,
        default=None,
        help="Factory IDs to download (e.g., '1-5,10,15-20')"
    )

    parser.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Worker IDs to download (e.g., '1-3,5')"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse ranges
    factories = parse_range(args.factories) if args.factories else None
    workers = parse_range(args.workers) if args.workers else None

    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run download
    logger.info(f"Output directory: {args.output.absolute()}")
    logger.info(f"Parallel downloads: {args.parallel}")

    if factories:
        logger.info(f"Factories filter: {factories}")
    if workers:
        logger.info(f"Workers filter: {workers}")

    start_time = time.time()

    asyncio.run(download_dataset(
        output_dir=args.output,
        max_concurrent=args.parallel,
        factories=factories,
        workers=workers,
        hf_token=hf_token,
        dry_run=args.dry_run
    ))

    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
