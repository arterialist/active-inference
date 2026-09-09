"""Separate PAULA sensorimotor-compass agent, version two.

This module is deliberately opt-in.  It keeps the working demonstration
agent and the earlier v1 residual experiment unchanged while correcting the
sensory representation discovered by their full-tick comparison.  See
``SENSORIMOTOR_COMPASS_DESIGN.md`` for the biological rationale, evidence
boundary, and versioned experimental results.
"""

from __future__ import annotations

from . import aif_agent3d as base


SENSORIMOTOR_COMPASS_V2_DEFAULTS = {
    "sensorimotor_estimator_v2": True,
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
    # Four ordinary PAULA time scales are a population representation of
    # movement uncertainty.  The physical interface supplies only unsigned
    # opposing yaw magnitudes; signed evidence is made by synaptic weights.
    "sensorimotor_v2_sensory_taus": (6.0, 12.0, 24.0, 48.0),
    "sensorimotor_v2_prediction_taus": (6.0, 12.0, 24.0, 48.0),
    "sensorimotor_v2_sensory_input_gain": 0.12,
    "sensorimotor_v2_sensory_weight": 1.0,
    "sensorimotor_v2_prediction_input_gain": 6.0,
    "sensorimotor_v2_prediction_weight": 1.0,
    "sensorimotor_v2_error_weight": 1.0,
    "sensorimotor_v2_update_gain": 1.0,
    "sensorimotor_v2_update_lam": 4.0,
    "sensorimotor_v2_pen_gain": 1.0,
}


class SensorimotorEstimatorV2Agent3D(base.AIFAgent3D):
    """Alternative all-PAULA heading-update candidate; defaults remain intact."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V2_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2"):
            raise ValueError("SensorimotorEstimatorV2Agent3D requires sensorimotor_estimator_v2=True")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV2Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v2",
            "sensory_self_motion": {
                "taus": tuple(SENSORIMOTOR_COMPASS_V2_DEFAULTS["sensorimotor_v2_sensory_taus"]),
                "source": "raw physical yaw split into non-negative opposing currents",
                "sign": "ordinary excitatory/inhibitory PAULA dendrites",
            },
            "motor_prediction": {
                "taus": tuple(SENSORIMOTOR_COMPASS_V2_DEFAULTS["sensorimotor_v2_prediction_taus"]),
                "source": "final spiking paddle relays",
            },
            "fusion": "opponent PAULA prediction error reconciles motor and self-motion evidence",
            "heading_state": "existing E-PG ring; local conjunctive P-EN update",
        }


run_episode = base.run_episode
