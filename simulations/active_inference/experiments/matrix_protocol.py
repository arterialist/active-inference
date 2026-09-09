"""Shared rules for the embodied version-acceptance matrix.

The matrix is deliberately a small, auditable contract rather than a score
that can be tuned after a run.  A protocol run contains ten declared worlds in
the order in which their challenge is introduced (simple -> hard), five fixed
replication seeds, and one common horizon derived from the hardest selected
world.  Exploratory lab runs may use a subset, but they are marked
``protocol_compliant=False`` and cannot be reported as version acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PROTOCOL_NAME = "embodied_matrix"
PROTOCOL_VERSION = "1.0"
WORLD_COUNT = 10
SEED_COUNT = 5
PROTOCOL_SEEDS = (11, 23, 44, 77, 101)


def world_names(world_specs: Mapping[str, Mapping]) -> tuple[str, ...]:
    """Return the catalog order; order is scientific metadata, not sorting."""
    return tuple(str(name) for name in world_specs)


def selected_worlds(world_specs: Mapping[str, Mapping], requested: str | None) -> list[str]:
    names = list(world_names(world_specs))
    if requested in (None, "all"):
        return names
    if requested not in world_specs:
        raise ValueError(f"unknown world {requested!r}")
    return [str(requested)]


def hardest_completion_steps(world_specs: Mapping[str, Mapping], worlds: Sequence[str]) -> int:
    """Return the horizon for a selected catalog slice.

    Every world declares ``completion_steps`` in the units used by its
    harness (body steps for embodied routes, trials for the memory route).
    Keeping that unit explicit avoids smuggling an arbitrary endpoint window
    into a protocol result.
    """
    if not worlds:
        raise ValueError("at least one world is required")
    values = []
    for name in worlds:
        try:
            value = int(world_specs[name]["completion_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"world {name!r} lacks a positive completion_steps value") from exc
        if value <= 0:
            raise ValueError(f"world {name!r} has non-positive completion_steps={value}")
        values.append(value)
    return max(values)


def protocol_compliance(
    *,
    world_specs: Mapping[str, Mapping],
    worlds: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    requested_world: str | None = None,
) -> tuple[bool, list[str]]:
    """Check the immutable ten-world/five-seed/horizon contract."""
    failures: list[str] = []
    catalog = list(world_names(world_specs))
    if len(catalog) != WORLD_COUNT:
        failures.append(f"world catalog has {len(catalog)} worlds, expected {WORLD_COUNT}")
    if list(worlds) != catalog:
        failures.append("protocol worlds are not the complete catalog in declared order")
    if requested_world not in (None, "all"):
        failures.append("protocol runs must select --world all")
    if tuple(int(seed) for seed in seeds) != PROTOCOL_SEEDS:
        failures.append(f"protocol seeds must be exactly {list(PROTOCOL_SEEDS)}")
    if len(seeds) != SEED_COUNT:
        failures.append(f"protocol requires {SEED_COUNT} seeds")
    try:
        hardest = hardest_completion_steps(world_specs, worlds)
    except ValueError as exc:
        failures.append(str(exc))
    else:
        if int(steps) < hardest:
            failures.append(f"horizon {steps} is shorter than hardest world completion {hardest}")
    return not failures, failures


def protocol_metadata(
    *,
    world_specs: Mapping[str, Mapping],
    worlds: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    required: bool,
    requested_world: str | None = None,
) -> dict:
    """Serialize the protocol decision into every child manifest."""
    compliant, failures = protocol_compliance(
        world_specs=world_specs,
        worlds=worlds,
        seeds=seeds,
        steps=steps,
        requested_world=requested_world,
    )
    hardest = None
    try:
        hardest = hardest_completion_steps(world_specs, worlds)
    except ValueError:
        pass
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "required": bool(required),
        "world_count": WORLD_COUNT,
        "seed_count": SEED_COUNT,
        "seeds": [int(seed) for seed in seeds],
        "worlds": list(worlds),
        "world_order": "catalog declaration (simple to hard)",
        "horizon_policy": "max(world_catalog[*].completion_steps)",
        "hardest_completion_steps": hardest,
        "horizon_steps": int(steps),
        "parallel": False,
        "compliant": compliant,
        "failures": failures,
    }


def require_protocol(
    *,
    world_specs: Mapping[str, Mapping],
    worlds: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    requested_world: str | None,
) -> None:
    compliant, failures = protocol_compliance(
        world_specs=world_specs,
        worlds=worlds,
        seeds=seeds,
        steps=steps,
        requested_world=requested_world,
    )
    if not compliant:
        raise ValueError("embodied matrix protocol violation: " + " | ".join(failures))

