"""Agent and environment contracts for constraint-driven co-evolution."""

from __future__ import annotations

from dataclasses import dataclass

from .components import ComponentRegistry, registry


@dataclass(frozen=True)
class ChallengeSpec:
    name: str
    prior_failure: str
    minimum_body_delta: tuple[str, ...]
    required_neural_delta: tuple[str, ...]
    acceptance_harness: str
    ablation_harness: str | None = None


@dataclass(frozen=True)
class AgentBlueprint:
    name: str
    inherited: tuple[str, ...]
    added: tuple[str, ...] = ()
    challenge: ChallengeSpec | None = None

    @property
    def components(self) -> tuple[str, ...]:
        return self.inherited + self.added

    def validate(self, component_registry: ComponentRegistry = registry) -> None:
        component_registry.validate(self.components)
        if self.challenge is not None:
            if not set(self.challenge.required_neural_delta) <= set(self.added):
                raise ValueError("A challenge's neural delta must be explicit in this version's added components")
