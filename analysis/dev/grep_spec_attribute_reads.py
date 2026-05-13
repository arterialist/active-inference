"""Search the repo for reads of selected spec-backed attributes (``grep``)."""
from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SEARCH_ROOT = _REPO / "simulations"

dead_candidates = [
    ("_gap_diffusion_rate", "sim.thermo.gap_diffusion_rate"),
    ("_tonic_metabolic_heat", "sim.thermo.tonic_metabolic_heat"),
    ("_nmj_scale", "sim.muscles.nmj_scale"),
    ("_nmj_threshold", "sim.muscles.nmj_threshold"),
]


def main() -> None:
    for attr, path in dead_candidates:
        # Search in the codebase (excluding parameter files themselves)
        result = subprocess.run(
            f"grep -r --exclude-dir=__pycache__ --exclude='*.npz' '{attr}' {_SEARCH_ROOT} 2>/dev/null "
            f"| grep -v 'simulation_params.py' | grep -v test | grep -v '.pyc'",
            shell=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        lines = [l for l in lines if l]

        print(f"\n{attr} (from {path}):")
        if not lines:
            print("  ✗ DEAD: no reads found in codebase (only setter in parameters.py)")
        else:
            for line in lines[:5]:
                print(f"  {line}")
            if len(lines) > 5:
                print(f"  ... and {len(lines) - 5} more")


if __name__ == "__main__":
    main()
