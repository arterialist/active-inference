"""Sensory encoding for the reduced larval zebrafish circuit."""

from __future__ import annotations

import numpy as np

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.zebrafish import config as zfc


class ZebrafishSensorEncoder:
    """Flatten environment/body observations into named PAULA sensory inputs."""

    def encode(
        self,
        obs: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        extra = obs.extra
        chemicals = obs.chemicals
        inputs = {
            "RETINA_L": float(extra.get("visual_left", 0.0)),
            "RETINA_R": float(extra.get("visual_right", 0.0)),
            "OPTIC_FLOW_L": float(extra.get("optic_flow_left", 0.0)),
            "OPTIC_FLOW_R": float(extra.get("optic_flow_right", 0.0)),
            "LATERAL_LINE_L": float(extra.get("lateral_line_left", 0.0)),
            "LATERAL_LINE_R": float(extra.get("lateral_line_right", 0.0)),
            "THERMO_HOT": float(extra.get("thermo_hot", 0.0)),
            "THERMO_COLD": float(extra.get("thermo_cold", 0.0)),
            "WALL_L": float(extra.get("wall_left", 0.0)),
            "WALL_R": float(extra.get("wall_right", 0.0)),
            "DEPTH_SHALLOW": float(extra.get("depth_shallow", 0.0)),
            "DEPTH_DEEP": float(extra.get("depth_deep", 0.0)),
            "STARTLE": float(extra.get("startle", 0.0)),
            "VIDEO_ACTIVE": float(extra.get("video_active", 0.0)),
            "VIDEO_MOTION": float(extra.get("video_motion_energy", 0.0)),
            "VIDEO_ASYMMETRY": float(extra.get("video_asymmetry", 0.0)),
            "VIDEO_ACTION_KICK": float(extra.get("video_action_kick", 0.0)),
            "VIDEO_ACTION_FORCE": float(extra.get("video_action_force", 0.0)),
            "VIDEO_ACTION_SIDE_SCORE": float(extra.get("video_action_side_score", 0.0)),
            "VIDEO_ACTION_CONFIDENCE": float(extra.get("video_action_confidence", 0.0)),
            "CALCIUM_ACTIVE": float(extra.get("calcium_active", 0.0)),
            "CALCIUM_KICK": float(extra.get("calcium_kick", 0.0)),
            "CALCIUM_FORCE": float(extra.get("calcium_force", 0.0)),
            "CALCIUM_SIDE_SCORE": float(extra.get("calcium_side_score", 0.0)),
            "CALCIUM_CONFIDENCE": float(extra.get("calcium_confidence", 0.0)),
            "ACTION_ACTIVE": float(extra.get("action_active", 0.0)),
            "ACTION_KICK": float(extra.get("action_kick", 0.0)),
            "ACTION_FORCE": float(extra.get("action_force", 0.0)),
            "ACTION_SIDE_SCORE": float(extra.get("action_side_score", 0.0)),
            "ACTION_CONFIDENCE": float(extra.get("action_confidence", 0.0)),
            "ACTION_TAIL_TARGET_CONFIDENCE": float(extra.get("action_tail_target_confidence", 0.0)),
        }
        for i in range(zfc.N_BODY_SEGMENTS):
            inputs[f"ACTION_TAIL_TARGET_{i:02d}"] = float(
                np.clip(extra.get(f"action_tail_target_{i:02d}", 0.0), -1.0, 1.0)
            )
        odor = float(chemicals.get("food_odor", 0.0))
        bearing = float(extra.get("target_bearing_rad", 0.0))
        side = np.sin(bearing)
        inputs["OLFACTORY_L"] = float(np.clip(odor * (0.65 + 0.35 * max(0.0, side)), 0.0, 1.0))
        inputs["OLFACTORY_R"] = float(np.clip(odor * (0.65 + 0.35 * max(0.0, -side)), 0.0, 1.0))

        yaw_rate = float(body_state.extra.get("angular_velocity_rad_s", 0.0))
        inputs["VESTIBULAR_L"] = float(np.clip(max(0.0, yaw_rate) / 8.0, 0.0, 1.0))
        inputs["VESTIBULAR_R"] = float(np.clip(max(0.0, -yaw_rate) / 8.0, 0.0, 1.0))
        vertical_velocity = float(body_state.extra.get("vertical_velocity_m_s", 0.0))
        inputs["VESTIBULAR_UP"] = float(np.clip(max(0.0, vertical_velocity) / 0.025, 0.0, 1.0))
        inputs["VESTIBULAR_DOWN"] = float(np.clip(max(0.0, -vertical_velocity) / 0.025, 0.0, 1.0))
        return inputs
