"""Phase-distributed leading-update variant of the separate PAULA compass.

V3 keeps V2's corrected, explicit opponent sensory representation and its
separate relay prediction.  It replaces only the final static fused-update
projection with an ordinary PAULA phase-distributed coincidence tract.  This
is an experimental P-ENa-like leading route, not a literal fly connectome or
a claim that the following compass is solved.
"""

from __future__ import annotations

from . import aif_agent3d as base
from .sensorimotor_agent3d_v2 import SENSORIMOTOR_COMPASS_V2_DEFAULTS


SENSORIMOTOR_COMPASS_V3_DEFAULTS = {
    **SENSORIMOTOR_COMPASS_V2_DEFAULTS,
    # The V2 full-tick trace showed that a 44-tick ordinary membrane removes
    # opponent gait flicker.  The new phase tract adds two neural transport
    # stages after the fused update, so d_push=2 matches the direct P-EN
    # mechanism's four-tick delay rather than silently changing wave phase.
    "sensorimotor_v2_update_lam": 44.0,
    "sensorimotor_v2_phase_gated_lead": True,
    "sensorimotor_v2_phase_copies": 6,
    "sensorimotor_v2_phase_gate_gain": 1.0,
    "d_push": 2,
}


class SensorimotorEstimatorV3Agent3D(base.AIFAgent3D):
    """Opt-in V3 leading-update hypothesis; the demo topology is unchanged."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V3_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2"):
            raise ValueError("SensorimotorEstimatorV3Agent3D requires sensorimotor_estimator_v2=True")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV3Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v3_phase_lead",
            "sensory_self_motion": "V2 non-negative opponent current ports and PAULA inhibitory dendrites",
            "motor_prediction": "V2 final-spiking-relay population",
            "fusion": "V2 paired PAULA prediction-error populations",
            "leading_update": {
                "form": "four CPG phases × six ordinary PAULA coincidence cells per direction",
                "source": "fused update × CPG phase, then local E-PG-conjunctive P-EN",
                "claim": "experimental P-ENa-like timing hypothesis; not a literal subtype count",
            },
            "trailing_stabilizer": "not yet present; P-EG/P-ENb remains a separate causal test",
        }


run_episode = base.run_episode
