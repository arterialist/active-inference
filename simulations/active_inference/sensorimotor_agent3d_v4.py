"""Leaky phase-population extension of the separate PAULA compass candidate.

This variant follows the V3 full-tick failure: single-tick CPG coincidences
made the leading update sparse and caused a jittering E-PG response. V4 uses
only inherited, first-order PAULA dendritic state to overlap adjacent phase
opportunities. It remains a falsifiable circuit hypothesis, not a special
sample/hold neuron or a biological-completeness claim.
"""

from __future__ import annotations

from . import aif_agent3d as base
from .sensorimotor_agent3d_v3 import SENSORIMOTOR_COMPASS_V3_DEFAULTS


SENSORIMOTOR_COMPASS_V4_DEFAULTS = {
    **SENSORIMOTOR_COMPASS_V3_DEFAULTS,
    # Four cells per motor phase give a predicted aggregate leading drive
    # close to the isolated P-EN transfer band's 0.5 current when four-tick
    # first-order phase traces overlap. This is a population-size hypothesis,
    # not a hidden gain calculation during simulation.
    "sensorimotor_v2_phase_copies": 4,
    "sensorimotor_v2_phase_gate_tau": 4.0,
}


class SensorimotorEstimatorV4Agent3D(base.AIFAgent3D):
    """Opt-in leaky phase-population candidate; default agent remains unchanged."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V4_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2"):
            raise ValueError("SensorimotorEstimatorV4Agent3D requires sensorimotor_estimator_v2=True")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV4Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v4_leaky_phase_population",
            "sensory_self_motion": "V2 non-negative opponent current ports and PAULA inhibitory dendrites",
            "motor_prediction": "V2 final-spiking-relay population",
            "fusion": "V2 paired PAULA prediction-error populations",
            "leading_update": {
                "form": "four CPG phases × four ordinary PAULA coincidence cells per direction",
                "local_dynamics": "four-tick first-order conjunctive phase-dendrite trace",
                "claim": "experimental P-ENa-like timing/population hypothesis; not a literal subtype count",
            },
            "trailing_stabilizer": "not yet present; P-EG/P-ENb remains a separate causal test",
        }


run_episode = base.run_episode
