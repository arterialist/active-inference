"""
C. elegans neuromuscular junction model.

Converts the per-muscle activation dict produced by CElegansNervousSystem
into the ctrl array format expected by CElegansBody.step().

The mapping is straightforward (activation -> ctrl) but this module also
exposes utility functions for analysing the motor pattern.
"""

from __future__ import annotations

import numpy as np

from simulations.c_elegans.config import (
    N_BODY_SEGMENTS,
    MUSCLE_QUADRANTS,
    MUSCLE_FILTER_ALPHA,
)


class NeuromuscularJunction:
    """
    Stateless mapper: per-muscle activations -> MuJoCo ctrl dict.

    MuJoCo expects one ctrl value per actuator.  The actuators in
    body_model.xml are named ``muscle_seg{N}_{QUAD}`` where N is 1-12
    (segment index) and QUAD ∈ {DL, DR, VL, VR}.

    The nervous system internally uses ``seg{N}_{QUAD}`` (0-indexed).
    This class resolves the 0-indexed -> 1-indexed offset.
    """

    @staticmethod
    def to_ctrl(activations: dict[str, float]) -> dict[str, float]:
        """
        Convert nervous-system activations to MuJoCo actuator names.

        Args:
            activations: {seg{N}_{QUAD}: float in [0,1]} from nervous system.

        Returns:
            {muscle_seg{N+1}_{QUAD}: float in [0,1]} for body.step().
        """
        ctrl: dict[str, float] = {}
        for key, val in activations.items():
            if not key.startswith("seg"):
                continue
            # Parse "seg{N}_{QUAD}"
            rest = key[3:]          # "{N}_{QUAD}"
            underscore = rest.index("_")
            seg_idx = int(rest[:underscore])
            quad = rest[underscore + 1:]

            if 0 <= seg_idx < N_BODY_SEGMENTS and quad in MUSCLE_QUADRANTS:
                mj_name = f"muscle_seg{seg_idx + 1}_{quad}"
                ctrl[mj_name] = float(np.clip(val, 0.0, 1.0))

        return ctrl

    @staticmethod
    def dorsal_minus_ventral(activations: dict[str, float]) -> np.ndarray:
        """
        Return (N_BODY_SEGMENTS,) array of (dorsal - ventral) activation.

        Positive values indicate dorsal contraction → body bends dorsally.
        Useful for visualising the travelling wave.
        """
        diff = np.zeros(N_BODY_SEGMENTS)
        for seg in range(N_BODY_SEGMENTS):
            d = (
                activations.get(f"seg{seg}_DL", 0.0)
                + activations.get(f"seg{seg}_DR", 0.0)
            ) / 2.0
            v = (
                activations.get(f"seg{seg}_VL", 0.0)
                + activations.get(f"seg{seg}_VR", 0.0)
            ) / 2.0
            diff[seg] = d - v
        return diff

    @staticmethod
    def mean_activation(activations: dict[str, float]) -> float:
        """Average muscle activation across all muscles."""
        vals = list(activations.values())
        return float(np.mean(vals)) if vals else 0.0
