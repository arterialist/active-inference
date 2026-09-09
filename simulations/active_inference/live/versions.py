"""The strict version contract shared by the live server, exporter, and lab UI.

Each tuple is passed into the PAULA builder as a topology profile.  A version
therefore emits only its declared populations; it never hides an inherited
full-brain graph behind a decoder mask or swaps in a Python policy.
"""

from __future__ import annotations

from dataclasses import dataclass


_BASE = (
    "sensory.olfactory_valence",
    "motor.cpg_muscle",
)
_MEMORY = _BASE + ("learning.mushroom_body",)
_V3 = _MEMORY + (
    "arbitration.foraging_exploration",
    "body.metabolic_organs",
    "arbitration.metabolic_sleep",
)
_V4 = _V3 + (
    "body.obstacle_geometry",
    "sensory.obstacle_proximity",
    "motor.obstacle_reflex",
)


@dataclass(frozen=True)
class AgentVersion:
    id: str
    label: str
    description: str
    components: tuple[str, ...]
    neural_delta: tuple[str, ...] = ()
    body_delta: tuple[str, ...] = ()
    # A topology existing is not behavioural acceptance. Promotion requires
    # current, complete embodied evidence under the version protocol.
    accepted: bool = False

    def manifest(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "components": list(self.components),
            "neural_delta": list(self.neural_delta),
            "body_delta": list(self.body_delta),
            "accepted": self.accepted,
        }


VERSIONS = {
    "v1": AgentVersion(
        "v1", "Reactive V1",
        "Reactive olfactory/toxin behaviour on the established PAULA CPG and graded-muscle body.",
        _BASE,
    ),
    "v2": AgentVersion(
        "v2", "Memory V2",
        "V1 plus the PAULA mushroom-body learned-valence composition.",
        _MEMORY,
        neural_delta=("learning.mushroom_body",),
    ),
    "v3": AgentVersion(
        "v3", "Interoceptive V3",
        "V2 plus metabolic organs and a PAULA FORAGE/EXPLORE/SLEEP arbiter.",
        _V3,
        neural_delta=("arbitration.foraging_exploration", "arbitration.metabolic_sleep"),
        body_delta=("body.metabolic_organs",),
    ),
    "v4": AgentVersion(
        "v4", "Tactile-detour V4",
        "V3 plus deterministic wall/corner/chicane/maze challenges, bilateral whisker proximity, and a PAULA obstacle reflex.",
        _V4,
        neural_delta=("sensory.obstacle_proximity", "motor.obstacle_reflex"),
        body_delta=("body.obstacle_geometry",),
    ),
}

_ALIASES = {"1": "v1", "2": "v2", "3": "v3", "4": "v4", "reactive": "v1", "memory": "v2", "interoceptive": "v3", "tactile": "v4", "obstacle": "v4"}


def get_version(value: str | AgentVersion) -> AgentVersion:
    if isinstance(value, AgentVersion):
        return value
    key = str(value).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return VERSIONS[key]
    except KeyError as exc:
        raise ValueError(f"unknown agent version {value!r}; choose one of {', '.join(VERSIONS)}") from exc


def version_ids() -> tuple[str, ...]:
    return tuple(VERSIONS)
