"""Stable composition contracts for the custom embodied PAULA organism."""

from .components import ComponentSpec, ComponentStatus, registry
from .composition import AgentBlueprint, ChallengeSpec

__all__ = ["AgentBlueprint", "ChallengeSpec", "ComponentSpec", "ComponentStatus", "registry"]
