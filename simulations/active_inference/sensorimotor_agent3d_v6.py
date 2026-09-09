"""Separate sensorimotor compass with a P-EG -> P-ENb trailing route.

V6 is an opt-in successor to V5.  It removes V5's harmful direct
same-column P-EG -> E-PG return and instead builds a distinct pair of ordinary
PAULA P-ENb-like cells per column.  Each has one local P-EG dendrite and one
direction-selecting dendrite from V2's PAULA sensory/motor fused update.  The
P-ENb outputs return at the opposite spatial offset to the early P-ENa-like
route.  This is a testable network hypothesis informed by the separate
P-EN/P-EG recurrent loops reported in the fly central complex; it is not an
anatomical reconstruction or an accepted navigation configuration.
"""

from __future__ import annotations

from . import aif_agent3d as base
from .sensorimotor_agent3d_v2 import SENSORIMOTOR_COMPASS_V2_DEFAULTS


SENSORIMOTOR_COMPASS_V6_DEFAULTS = {
    **SENSORIMOTOR_COMPASS_V2_DEFAULTS,
    # The slow fused opponent state rejected much of the alternating gait
    # common-mode signal in V2.  Its leading route remains present unchanged.
    "sensorimotor_v2_update_lam": 44.0,
    "d_push": 3,
    # Explicit PB/P-EG relay, but no direct P-EG -> same-column E-PG loop.
    "pb_eb_bridge": True,
    "w_peg": 1.6,
    "w_pb_peg": 1.6,
    "w_peg_ring": 0.0,
    # New distinct trailing route.  Both local P-EG and signed fused-update
    # drive are individually subthreshold at the declared default weights.
    "p_enb_trailing": True,
    "lam_penb": 2.0,
    "w_peg_penb": 0.6,
    "penb_conjunctive_gain": 1.0,
    "penb_ring_tau": 4.0,
    "penb_velocity_tau": 4.0,
    "p_enb_update_gain": 3.0,
    "w_penb_ring": 0.12,
    "d_penb": 3,
}


class SensorimotorEstimatorV6Agent3D(base.AIFAgent3D):
    """Opt-in two-route P-ENa/P-ENb-like compass hypothesis."""

    def __init__(self, *args, **kwargs):
        build = dict(SENSORIMOTOR_COMPASS_V6_DEFAULTS)
        build.update(kwargs)
        if not build.get("sensorimotor_estimator_v2"):
            raise ValueError("SensorimotorEstimatorV6Agent3D requires sensorimotor_estimator_v2=True")
        if not (build.get("pb_eb_bridge") and build.get("w_peg") and build.get("p_enb_trailing")):
            raise ValueError("SensorimotorEstimatorV6Agent3D requires PB/P-EG and P-ENb routes")
        if build.get("sensorimotor_estimator"):
            raise ValueError("SensorimotorEstimatorV6Agent3D cannot enable v1 simultaneously")
        super().__init__(*args, **build)

    def sensorimotor_manifest(self) -> dict:
        return {
            "agent_variant": "sensorimotor_estimator_v6_two_route_trailing",
            "sensory_self_motion": "V2 non-negative opponent PAULA current ports",
            "motor_prediction": "V2 final-relay PAULA efference population",
            "leading_update": "fused update -> P-ENa-like PB offset -> E-PG",
            "trailing_update": {
                "form": "E-PG -> bilateral PB -> P-EG -> signed P-ENb -> opposite-offset E-PG",
                "sources": "local P-EG plus the same PAULA fused self-motion signal as the leading route",
                "claim": "experimental timing/spatial-topology hypothesis; not a connectome reconstruction",
            },
            "excluded": ["host heading integration", "pose-derived heading", "direct P-EG same-column return"],
        }


run_episode = base.run_episode
