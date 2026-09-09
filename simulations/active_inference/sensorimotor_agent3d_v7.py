"""Separate scaled P-ENa population experiment for embodied sensorimotor yaw.

V7 tests the user's scaling hypothesis directly.  It preserves V2's corrected
opponent sensory/motor fusion and the ordinary leading P-EN route, then adds
four distinct P-ENa-like PAULA cells per column per direction.  Their local
velocity dendrites have time constants 1, 2, 4, and 8 ticks, so the population
can tolerate small transport/gait phase shifts without introducing a stroke
counter, sample/hold mechanism, decoded heading, or host-side smoother.

This is an opt-in engineering/neuroscience hypothesis, not a claim that a
specific animal uses exactly four such cells or these constants.
"""

from __future__ import annotations

from . import aif_agent3d as base
from .sensorimotor_agent3d_v2 import SENSORIMOTOR_COMPASS_V2_DEFAULTS


SENSORIMOTOR_COMPASS_V7_DEFAULTS = {
    **SENSORIMOTOR_COMPASS_V2_DEFAULTS,
    "sensorimotor_v2_update_lam": 44.0,
    "d_push": 3,
    # Extra P-ENa cells retain the existing route and add only a modest
    # parallel leading conductance.  Copy-specific local dynamics, rather
    # than a single multiplied terminal, are the experimental variable.
    "p_ena_bank": True,
    "pena_copies": 4,
    "pena_gain": 0.5,
    "pena_ring_tau": 1.0,
    "pena_velocity_taus": (1.0, 2.0, 4.0, 8.0),
    "p_ena_update_gain": 1.0,
    "w_pena_ring": 0.35,
    "d_pena": 3,
}


class SensorimotorEstimatorV7Agent3D(base.AIFAgent3D):
    """Opt-in heterogeneous P-ENa population-scale candidate."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V7_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2") or not build.get("p_ena_bank"):
            raise ValueError("SensorimotorEstimatorV7Agent3D requires V2 fusion and the P-ENa micro-bank")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV7Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v7_heterogeneous_pena_bank",
            "base": "V2 non-negative opponent sensory/motor fusion",
            "leading_update": {
                "form": "existing local P-EN plus four local P-ENa-like PAULA copies per column/direction",
                "velocity_taus": SENSORIMOTOR_COMPASS_V7_DEFAULTS["pena_velocity_taus"],
                "claim": "experimental population scale-up, not an anatomical count claim",
            },
            "excluded": ["host smoothing", "phase counter", "sample/hold", "decoded heading feedback"],
        }


run_episode = base.run_episode
