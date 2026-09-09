"""Declarative inventory of reusable organism components.

This registry is deliberately capability-based, not a closed anatomical
taxonomy.  A component may live in any package while declaring what body
signals it needs, what it exposes, its dependencies, and the acceptance
harness that establishes its present status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ComponentStatus(str, Enum):
    ACCEPTED = "accepted"
    EXPERIMENTAL = "experimental"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    module: str
    status: ComponentStatus
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    acceptance_harness: str | None = None
    notes: str = ""


@dataclass
class ComponentRegistry:
    _specs: dict[str, ComponentSpec] = field(default_factory=dict)

    def register(self, spec: ComponentSpec) -> ComponentSpec:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate component: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ComponentSpec:
        return self._specs[name]

    def all(self) -> Mapping[str, ComponentSpec]:
        return dict(self._specs)

    def validate(self, names: tuple[str, ...]) -> None:
        selected = set(names)
        unknown = selected - set(self._specs)
        if unknown:
            raise ValueError(f"Unknown component(s): {', '.join(sorted(unknown))}")
        for name in selected:
            spec = self._specs[name]
            missing = set(spec.dependencies) - selected
            if missing:
                raise ValueError(f"{name} requires component(s): {', '.join(sorted(missing))}")
            incompatible = set(spec.conflicts) & selected
            if incompatible:
                raise ValueError(f"{name} conflicts with: {', '.join(sorted(incompatible))}")


registry = ComponentRegistry()

registry.register(ComponentSpec(
    "sensory.olfactory_valence", "components.sensory.olfactory", ComponentStatus.ACCEPTED,
    requires=("body.odour_field",), provides=("food_gradient", "toxin_gradient"),
    acceptance_harness="experiments/embodied_food_gradient_causal.py",
))
registry.register(ComponentSpec(
    "motor.cpg_muscle", "components.motor.cpg", ComponentStatus.ACCEPTED,
    requires=("body.mujoco_muscles",), provides=("rhythmic_locomotion",),
    acceptance_harness="experiments/paula_motor_causal.py",
    notes="Retained established PAULA CPG/relay/graded-muscle route.",
))
registry.register(ComponentSpec(
    "learning.mushroom_body", "components.learning.mushroom_body", ComponentStatus.ACCEPTED,
    requires=("body.odour_field", "body.contact_us"), provides=("learned_valence",),
    dependencies=("sensory.olfactory_valence",),
    acceptance_harness="experiments/embodied_mb_valence_causal.py",
))
registry.register(ComponentSpec(
    "arbitration.foraging_exploration", "components.arbitration.interoceptive", ComponentStatus.ACCEPTED,
    requires=("hunger", "learned_valence"), provides=("forage_explore_selection",),
    acceptance_harness="experiments/embodied_v3_trajectory_causal.py",
))
registry.register(ComponentSpec(
    "body.metabolic_organs", "components.body.metabolism", ComponentStatus.ACCEPTED,
    requires=("body.food_contact",), provides=("gut_load", "energy_store", "metabolic_state"),
    acceptance_harness="experiments/embodied_metabolic_rest_causal.py",
    notes="V3 body transducer: physiology produces sensory afferents; it does not choose actions.",
))
registry.register(ComponentSpec(
    "body.obstacle_geometry", "components.body.obstacles", ComponentStatus.ACCEPTED,
    requires=("body.mujoco_geometry",), provides=("obstacle_distance", "obstacle_contact"),
    acceptance_harness="experiments/embodied_obstacle_detour_causal.py",
    notes="V4 body transducer: barrier geometry is physical; proximity/contact are measured afferents.",
))
registry.register(ComponentSpec(
    "sensory.obstacle_proximity", "components.sensory.obstacle", ComponentStatus.ACCEPTED,
    requires=("obstacle_distance", "obstacle_contact"), provides=("bilateral_obstacle_onset",),
    dependencies=("body.obstacle_geometry",),
    acceptance_harness="experiments/embodied_obstacle_detour_causal.py",
    notes="Bilateral whisker/range populations and delayed onset cells; no action selection.",
))
registry.register(ComponentSpec(
    "motor.obstacle_reflex", "components.motor.obstacle_reflex", ComponentStatus.ACCEPTED,
    requires=("bilateral_obstacle_onset", "body.mujoco_muscles"),
    provides=("collision_avoidance", "wall_following"),
    dependencies=("sensory.obstacle_proximity", "motor.cpg_muscle"),
    acceptance_harness="experiments/embodied_obstacle_detour_causal.py",
    notes="PAULA opponent turn/stop route converging on the established TL/TR and graded relays.",
))
registry.register(ComponentSpec(
    "arbitration.metabolic_sleep", "components.arbitration.metabolic_parts", ComponentStatus.ACCEPTED,
    requires=("gut_load", "energy_store", "uncertainty"), provides=("forage_explore_sleep_selection",),
    dependencies=("body.metabolic_organs", "arbitration.foraging_exploration"),
    acceptance_harness="experiments/embodied_metabolic_rest_causal.py",
    notes="V3 PAULA WTA extension; accepted after two-seed embodied ablation harness.",
))
registry.register(ComponentSpec(
    "navigation.heading_ring", "components.navigation.compass", ComponentStatus.EXPERIMENTAL,
    requires=("body.yaw_rate",), provides=("heading_belief",),
    acceptance_harness="experiments/embodied_compass_gyro_causal.py",
    notes="Alive and causally driven but not accepted for full embodied turn tracking.",
))
registry.register(ComponentSpec(
    "navigation.path_integration", "components.navigation.path_integration", ComponentStatus.QUARANTINED,
    requires=("heading_belief", "body.speed"), provides=("home_vector",),
    dependencies=("navigation.heading_ring",),
    acceptance_harness="experiments/embodied_xacc_pi_causal.py",
))
registry.register(ComponentSpec(
    "navigation.cylinder_compass", "components.navigation.cylinder", ComponentStatus.QUARANTINED,
    requires=("body.yaw_rate",), provides=("multi_scale_heading_belief",),
    acceptance_harness="../../experiments/embodied_cylinder_compass.py",
    notes="Retained parallel hypothesis; isolated behaviour is stronger than embodied transfer.",
))
