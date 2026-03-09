#!/usr/bin/env python3
"""
One-shot script to download and cache the C. elegans connectome.

Run this before starting simulations to pre-populate the cache:

    cd active-inference/
    uv run python scripts/download_connectome.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.c_elegans.connectome import load_connectome, print_connectome_summary

if __name__ == "__main__":
    print("Downloading / parsing C. elegans connectome …")
    data = load_connectome(use_cache=False)  # Force fresh parse
    print_connectome_summary(data)
    print("Connectome cached successfully.")
