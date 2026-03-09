"""
PAULA neuron model loader.

Centralizes the path setup required to import the PAULA neuron model
from the sibling neuron-model repository. Call ensure_paula_available()
before any neuron imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PAULA_PATH = Path(__file__).resolve().parents[2] / "neuron-model"


def ensure_paula_available() -> Path:
    """
    Ensure the neuron-model package is on sys.path for PAULA imports.

    Returns:
        Path to the neuron-model directory.
    """
    path_str = str(_PAULA_PATH.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return _PAULA_PATH.resolve()
