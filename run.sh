#!/bin/bash
# Setup and run the Egocentric-10K downloader

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."

    # Try uv first (fastest), then python venv
    if command -v uv &> /dev/null; then
        uv venv
        uv pip install aiohttp aiofiles tqdm
    elif command -v python3 &> /dev/null; then
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    else
        echo "Error: Neither uv nor python3 found"
        exit 1
    fi
fi

# Run the downloader with all arguments passed through
source .venv/bin/activate
exec python download_ego10k.py "$@"
