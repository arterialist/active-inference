"""Separate sensorimotor compass with an explicit P-EG/P-ENb-like stabilizer.

V5 combines V2's corrected sensory/motor fusion with the opt-in PB/EB
maintenance geometry already present in ``central_complex.py``.  The leading
update remains the local P-EN route; the trailing route is explicitly
E-PG -> bilateral PB maintenance tracts -> P-EG -> same-column E-PG.  It is
a structural hypothesis to test, not a claim that the topology or its weights
reconstruct a particular insect connectome.
"""

from __future__ import annotations

from . import aif_agent3d as base
from .sensorimotor_agent3d_v2 import SENSORIMOTOR_COMPASS_V2_DEFAULTS


SENSORIMOTOR_COMPASS_V5_DEFAULTS = {
    **SENSORIMOTOR_COMPASS_V2_DEFAULTS,
    # V2 full-tick evidence: avoid opponent half-stroke flicker and compensate
    # for one update-cell transport stage before the P-EN travelling wave.
    "sensorimotor_v2_update_lam": 44.0,
    "d_push": 3,
    # Separate P-EG/P-ENb-like maintenance loop through visible PB tracts.
    "pb_eb_bridge": True,
    "w_peg": 1.6,
    "w_pb_peg": 0.8,
    "w_peg_ring": 0.8,
    "d_peg": 1,
}


class SensorimotorEstimatorV5Agent3D(base.AIFAgent3D):
    """Opt-in leading-update plus trailing-stabilizer candidate."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V5_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2"):
            raise ValueError("SensorimotorEstimatorV5Agent3D requires sensorimotor_estimator_v2=True")
        if not build.get("pb_eb_bridge") or not build.get("w_peg"):
            raise ValueError("SensorimotorEstimatorV5Agent3D requires the explicit active PB/P-EG bridge")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV5Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v5_pb_peg_stabilizer",
            "sensory_self_motion": "V2 non-negative opponent current ports and PAULA inhibitory dendrites",
            "motor_prediction": "V2 final-spiking-relay population",
            "leading_update": "V2 fused update -> local conjunctive P-EN -> PB offset tract -> E-PG",
            "trailing_stabilizer": {
                "form": "E-PG -> bilateral PB maintenance tracts -> P-EG -> same-column E-PG",
                "claim": "experimental P-EG/P-ENb-like stabilizer; not an anatomical reconstruction",
            },
        }


run_episode = base.run_episode
