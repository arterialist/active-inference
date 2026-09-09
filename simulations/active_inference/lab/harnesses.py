"""Whitelisted embodied/isolated harnesses exposed by the web lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    label: str
    description: str
    script: str
    mode: str
    default_steps: int
    default_substeps: int
    default_seeds: tuple[int, ...]
    max_steps: int = 2000
    max_seeds: int = 5
    supported_versions: tuple[str, ...] = ("v1", "v2", "v3", "v4")
    worlds: tuple[str, ...] = ()
    default_world: str | None = None
    # ``strict_agent`` means the selected version is passed to the experiment
    # and the experiment must construct that exact component profile.  The two
    # other policies are deliberately visible in the lab because they are
    # useful, but they are not version acceptance claims: ``shared_component``
    # probes a lower-level route and ``legacy_experimental`` keeps an old
    # full-brain probe quarantined from the V1--V4 evidence table.
    version_policy: str = "strict_agent"
    pass_version: bool = True

    def command(self, *, output: Path, steps: int | None = None,
                substeps: int | None = None, seeds: list[int] | None = None,
                world: str | None = None, version: str | None = None) -> list[str]:
        # ``0`` is an invalid explicit request, not a signal to silently fall
        # back to the default.  The old truthiness shortcut made a malformed
        # form look accepted and launched a different experiment than the user
        # specified.
        steps = self.default_steps if steps is None else int(steps)
        substeps = self.default_substeps if substeps is None else int(substeps)
        seeds = list(self.default_seeds if seeds is None else seeds)
        if steps <= 0 or steps > self.max_steps:
            raise ValueError(f"steps must be in 1..{self.max_steps}")
        if substeps <= 0 or substeps > 64:
            raise ValueError("substeps must be in 1..64")
        if not seeds or len(seeds) > self.max_seeds or any(abs(int(s)) > 10_000_000 for s in seeds):
            raise ValueError(f"seeds must contain 1..{self.max_seeds} bounded integers")
        if self.worlds:
            world = self.default_world if world is None else str(world)
            if world not in self.worlds:
                raise ValueError(f"world must be one of {', '.join(self.worlds)}")
        version = self.supported_versions[0] if version is None and self.supported_versions else version
        if self.supported_versions and version not in self.supported_versions:
            raise ValueError(f"version must be one of {', '.join(self.supported_versions)}")
        command = [sys.executable, str(HERE / self.script)]
        if self.mode == "mb":
            command += ["--trials", str(min(32, steps)), "--train-steps", str(min(32, substeps)),
                        # The probe must remain long enough to contain the
                        # complete non-contact readout.  Using ``substeps``
                        # directly here made the web lab launch a four-tick
                        # probe by default, while the maintained acceptance
                        # contract assumes ten ticks; that discrepancy could
                        # turn the same harness into a different experiment.
                        "--probe-steps", str(min(64, max(10, substeps))), "--substeps", str(substeps)]
        else:
            command += ["--steps", str(steps), "--substeps", str(substeps)]
        if self.worlds:
            command += ["--world", str(world)]
        if self.pass_version and version is not None:
            command += ["--version", str(version)]
        command += ["--seeds", *[str(int(s)) for s in seeds], "--output", str(output)]
        return command


HARNESSES = {
    "motor": HarnessSpec(
        "motor", "CPG / graded muscle", "Open-loop PAULA CPG, graded muscles, NMJ transduction, and MuJoCo body.",
        "experiments/paula_motor_causal.py", "standard", 1800, 6, (11,), 1800, 4,
        version_policy="shared_component", pass_version=False,
    ),
    "food_collection": HarnessSpec(
        "food_collection", "Food sensor → turn", "Strict-version food route across ten ordered embodied target worlds.",
        "experiments/embodied_food_collection_causal.py", "standard", 300, 8, (11, 23, 44, 77, 101), 500, 5,
        version_policy="strict_agent", worlds=("shallow_left", "shallow_right", "near_left", "near_right", "mid_left", "mid_right", "far_left", "far_right", "deep_left", "deep_right", "all"),
        default_world="all",
    ),
    "toxin_escape": HarnessSpec(
        "toxin_escape", "Diagonal toxin escape", "Strict-version reactive hazard route across ten ordered embodied toxin worlds.",
        "experiments/embodied_toxin_escape_causal.py", "standard", 280, 8, (11, 23, 44, 77, 101), 500, 5,
        version_policy="strict_agent", worlds=("near_left", "near_right", "mid_left", "mid_right", "deep_left", "deep_right", "far_left", "far_right", "cross_left", "cross_right", "all"),
        default_world="all",
    ),
    "headon_toxin": HarnessSpec(
        "headon_toxin", "Head-on toxin / TRISE", "Strict-version temporal-rise escape across ten ordered symmetric hazards.",
        "experiments/embodied_headon_toxin_causal.py", "standard", 320, 8, (11, 23, 44, 77, 101), 500, 5,
        version_policy="strict_agent", worlds=("near_01", "near_02", "near_03", "mid_04", "mid_05", "mid_06", "far_07", "far_08", "far_09", "far_10", "all"),
        default_world="all",
    ),
    "memory": HarnessSpec(
        "memory", "Mushroom-body valence", "Physical toxin teaching followed by a non-contact learned-valence probe across ten offsets.",
        "experiments/embodied_mb_valence_causal.py", "mb", 8, 4, (11, 23, 44, 77, 101), 32, 5, ("v2", "v3"),
        version_policy="strict_agent", worlds=("center", "left_015", "right_015", "left_025", "right_025", "left_035", "right_035", "left_045", "right_045", "cross_050", "all"), default_world="all",
    ),
    "arbiter": HarnessSpec(
        "arbiter", "Forage / explore arbiter", "Hunger and physical food contact gating across ten meal-timing/heading worlds.",
        "experiments/embodied_arbiter_explore_causal.py", "standard", 128, 8, (11, 23, 44, 77, 101), 500, 5, ("v3",),
        version_policy="strict_agent", worlds=("early_0", "early_1", "early_2", "mid_3", "mid_4", "mid_5", "late_6", "late_7", "late_8", "late_9", "all"), default_world="all",
    ),
    "metabolic_rest": HarnessSpec(
        "metabolic_rest", "Metabolic SLEEP", "Gut, usable energy, digestion, and V3 SLEEP across ten meal delays.",
        "experiments/embodied_metabolic_rest_causal.py", "standard", 88, 4, (11, 23, 44, 77, 101), 300, 5, ("v3",),
        version_policy="strict_agent", worlds=("meal_04", "meal_06", "meal_08", "meal_10", "meal_12", "meal_14", "meal_16", "meal_18", "meal_20", "meal_24", "all"), default_world="all",
    ),
    "compass": HarnessSpec(
        "compass", "Embodied compass (experimental)",
        "Legacy full-brain compass probe; retained for research, but excluded from strict V1--V4 acceptance.",
        "experiments/embodied_compass_gyro_causal.py", "standard", 100, 8, (11,), 500, 4, ("v1", "v2", "v3"),
        version_policy="legacy_experimental", pass_version=False,
    ),
    "obstacle_detour": HarnessSpec(
        "obstacle_detour", "Tactile obstacle detour",
        "V4 geometry suite: full-width wall, L-corner, alternating chicane, and compact maze; no random food target.",
        "experiments/embodied_obstacle_detour_causal.py", "standard", 400, 4, (11, 23, 44, 77, 101), 600, 5,
        ("v4",),
        ("wall_short", "wall_medium", "wall_offset_left", "wall_offset_right", "head_on_wall", "corner_small", "corner", "chicane_short", "chicane", "maze", "all"), "all",
        version_policy="strict_agent",
    ),
}


def all_specs() -> list[dict]:
    return [{
        "id": s.id, "label": s.label, "description": s.description,
        "supported_versions": list(s.supported_versions), "default_steps": s.default_steps,
        "default_substeps": s.default_substeps, "default_seeds": list(s.default_seeds),
        "max_steps": s.max_steps, "mode": s.mode,
        "worlds": list(s.worlds), "default_world": s.default_world,
        "version_policy": s.version_policy, "pass_version": s.pass_version,
        "protocol": {
            "world_count": max(0, len(s.worlds) - (1 if s.worlds and s.worlds[-1] == "all" else 0)),
            "seed_count": 5 if s.version_policy == "strict_agent" else None,
            "horizon_policy": "max(world_catalog[*].completion_steps)" if s.version_policy == "strict_agent" else None,
        },
    } for s in HARNESSES.values()]
