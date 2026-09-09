"""V4 tactile-detour composition on the strict V3 topology."""

from __future__ import annotations

from ..interoceptive_v3.agent import InteroceptiveV3Agent, blueprint as v3_blueprint
from ...core import AgentBlueprint, ChallengeSpec


challenge = ChallengeSpec(
    name="obstacle_detour",
    prior_failure="V3 reaches a deterministic full-width physical wall but has no body channel for the barrier and remains blocked or collides.",
    minimum_body_delta=("body.obstacle_geometry",),
    required_neural_delta=("sensory.obstacle_proximity", "motor.obstacle_reflex"),
    acceptance_harness="experiments/embodied_obstacle_detour_causal.py",
    ablation_harness="experiments/embodied_obstacle_detour_causal.py",
)

blueprint = AgentBlueprint(
    name="obstacle_v4",
    inherited=v3_blueprint.components,
    added=("body.obstacle_geometry", "sensory.obstacle_proximity", "motor.obstacle_reflex"),
    challenge=challenge,
)
blueprint.validate()


class ObstacleV4Agent(InteroceptiveV3Agent):
    """One PAULA network with bilateral tactile detour circuitry."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("components", blueprint.components)
        super().__init__(*args, **kwargs)
