#!/usr/bin/env python3
"""
Apply evolved config from evolve_food_seeking.py to neuron_mapping defaults.

Usage:
  uv run python scripts/apply_evolved_config.py [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NM_PATH = Path(__file__).parent.parent / "simulations" / "c_elegans" / "neuron_mapping.py"

NEUROMOD_ATTRS = [
    "K_STRESS_SYN", "K_REWARD_SYN", "K_VOL_STRESS", "K_VOL_REWARD",
    "STRESS_DEADZONE", "CHEM_EMA_ALPHA_FAST", "CHEM_EMA_ALPHA_SLOW",
    "TONIC_FWD_CMD", "TONIC_FWD_MOTOR",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("evolved_food_seeking_config.json"))
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        print("Run evolve_food_seeking.py first.")
        sys.exit(1)

    with open(args.config) as f:
        data = json.load(f)
    # Support checkpoint format (from evolve_food_seeking --checkpoint)
    cfg = data.get("best_config", data)

    text = NM_PATH.read_text()

    for key in NEUROMOD_ATTRS:
        if key not in cfg:
            continue
        val = cfg[key]
        attr = f"_{key}"
        # Match: _ATTR: float = number
        pattern = re.escape(attr) + r": float = [\d.e+-]+"
        replacement = f"{attr}: float = {val}"
        text = re.sub(pattern, replacement, text, count=1)

    NM_PATH.write_text(text)
    print(f"Applied config to {NM_PATH}")


if __name__ == "__main__":
    main()
