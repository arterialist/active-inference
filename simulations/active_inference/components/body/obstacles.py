"""Body-side contract for the V4 obstacle transducer.

The actual geometry and measurement live in :mod:`components.body.world` so a
world cannot accidentally be replaced by a neural policy.  This small module
is intentionally first-class: it gives the component registry a stable body
capability name and documents the physical signal contract used by the PAULA
sensory fragment.
"""

from __future__ import annotations

from typing import Protocol


class ObstacleBody(Protocol):
    def obstacle_proximity(self) -> dict[str, float]:
        """Return physical left/right whisker-like proximity currents."""


OBSTACLE_SIGNAL_KEYS = (
    "left", "right", "left_onset", "right_onset",
    "distance_left", "distance_right", "contact",
)


def signal_contract() -> dict[str, object]:
    """Machine-readable body → brain contract for viewers and harnesses."""
    return {
        "component": "body.obstacle_geometry",
        "signals": list(OBSTACLE_SIGNAL_KEYS),
        "source": "MuJoCo barrier geometry and body pose",
        "policy": False,
    }
