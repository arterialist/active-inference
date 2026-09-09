"""V4's single environmental catalyst and its measurable contract."""

from __future__ import annotations


NAME = "obstacle_detour"
PRIOR_VERSION = "v3"
VERSION = "v4"
BODY_DELTA = ("body.obstacle_geometry",)
NEURAL_DELTA = ("sensory.obstacle_proximity", "motor.obstacle_reflex")
WORLD = "head_on_wall"


def manifest() -> dict[str, object]:
    return {
        "name": NAME,
        "prior_version": PRIOR_VERSION,
        "version": VERSION,
        "world": WORLD,
        "body_delta": list(BODY_DELTA),
        "neural_delta": list(NEURAL_DELTA),
        "acceptance": "experiments/embodied_obstacle_detour_causal.py",
    }
