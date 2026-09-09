"""Separate opt-in embodied agent for the sensorimotor compass programme.

The demonstration agent remains :class:`aif_agent3d.AIFAgent3D`.  This module
selects an alternative, all-PAULA heading-update architecture while retaining
the same body, chemosensation, mushroom body, arbiter, CPG and muscle path.
It deliberately makes no claim that the architecture is a literal insect
connectome.  See ``SENSORIMOTOR_COMPASS_DESIGN.md`` for the evidence boundary
and references.
"""

from __future__ import annotations

from . import aif_agent3d as base


# These defaults build the candidate only. They neither alter
# ``AIFAgent3D`` defaults nor the live-demo configuration. The update path is
# motor-relay prediction + raw-yaw residual -> PAULA prediction error -> local
# conjunctive P-EN -> existing E-PG ring.
SENSORIMOTOR_COMPASS_DEFAULTS = {
    "sensorimotor_estimator": True,
    "conjunctive_graded_shift": True,
    "conjunctive_graded_S0": 0.0,
    "graded_shift": True,
    "graded_shift_gain": 0.5,
    "graded_shift_S0": 0.0,
    "graded_ring": True,
    "graded_ring_gain": 0.08,
    "graded_ring_S0": 0.2,
    "d_push": 4,
    "w_push_ccw": 0.34,
    "w_push_cw": 0.67,
    "sensorimotor_sensory_windows": (36, 40, 44, 48),
    "sensorimotor_prediction_taus": (6.0, 12.0, 24.0, 48.0),
    "sensorimotor_sensory_input_gain": 1000.0,
    "sensorimotor_sensory_weight": 1.0,
    "sensorimotor_prediction_input_gain": 6.0,
    "sensorimotor_prediction_weight": 1.0,
    "sensorimotor_error_weight": 1.0,
    "sensorimotor_update_gain": 1.0,
    "sensorimotor_update_lam": 4.0,
    "sensorimotor_pen_gain": 1.0,
}


class SensorimotorEstimatorAgent3D(base.AIFAgent3D):
    """The alternate embodied agent; the normal demo class is unchanged."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorAgent3D requires sensorimotor_estimator=True")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        """Read-only description suitable for trace records and the live lab."""
        return {
            "agent_variant": "sensorimotor_estimator_v1",
            "sensory_residual": {
                "windows": tuple(SENSORIMOTOR_COMPASS_DEFAULTS["sensorimotor_sensory_windows"]),
                "source": "raw signed physical yaw current",
            },
            "motor_prediction": {
                "taus": tuple(SENSORIMOTOR_COMPASS_DEFAULTS["sensorimotor_prediction_taus"]),
                "source": "final spiking paddle relays",
            },
            "fusion": "PAULA sensory-minus-prediction residual plus motor prediction",
            "heading_state": "existing E-PG ring; local conjunctive P-EN update",
        }


run_episode = base.run_episode

