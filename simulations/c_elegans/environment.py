"""
C. elegans virtual environment.

Models a circular agar plate with:
  - A food/attractant source (e.g. diacetyl, NaCl, butanone)
  - An aversive odour source (2-nonanone)
  - A nociceptive (harmful touch) region
  - Flat surface for crawling locomotion

Chemical concentrations follow an exponential gradient:
    C(r) = C_max * exp(-λ * r)
where r is Euclidean distance from the source and λ is a decay constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simulations.base_environment import BaseEnvironment, EnvironmentObservation
from simulations.c_elegans.config import (
    ENV_PLATE_RADIUS_M,
    FOOD_SOURCE_POSITION,
    FOOD_GRADIENT_DECAY,
    CHEM_CONCENTRATION_MAX,
    FOOD_CONSUMPTION_RADIUS_M,
)


@dataclass
class ChemSource:
    """A point source of a named chemical molecule."""

    molecule: str
    position: np.ndarray   # 3-D (x, y, z) in biological metres
    max_concentration: float = 1.0
    decay_constant: float = FOOD_GRADIENT_DECAY
    # 'attractive' or 'aversive' — informational only
    valence: str = "attractive"

    def concentration_at(self, pos: np.ndarray) -> float:
        """Concentration at a given 3-D position."""
        r = float(np.linalg.norm(pos - self.position))
        return float(self.max_concentration * np.exp(-self.decay_constant * r))


class AgarPlateEnvironment(BaseEnvironment):
    """
    Circular agar plate environment.

    Args:
        food_position:   (x, y, z) of single food source in biological metres.
        food_positions:  List of (x, y, z) food sources. Overrides food_position if given.
        plate_radius:    Plate radius in metres.
        add_nociceptive: Whether to include a nociceptive (hot) region.
        noci_center:     Centre of nociceptive region.
        noci_radius:     Radius of nociceptive zone in metres.
    """

    def __init__(
        self,
        food_position: tuple[float, float, float] = FOOD_SOURCE_POSITION,
        food_positions: list[tuple[float, float, float]] | None = None,
        plate_radius: float = ENV_PLATE_RADIUS_M,
        add_nociceptive: bool = True,
        noci_center: tuple[float, float, float] = (-0.02, 0.0, 0.0),
        noci_radius: float = 0.005,
    ):
        self._plate_radius = plate_radius
        self._noci_center = np.array(noci_center)
        self._noci_radius = noci_radius

        positions = (
            food_positions
            if food_positions is not None
            else [food_position]
        )

        # Food items: (position, [NaCl, butanone] sources) — removed when worm touches
        self._food_items: list[tuple[np.ndarray, list[ChemSource]]] = []
        for pos in positions:
            pos_arr = np.array(pos)
            sources = [
                ChemSource(
                    molecule="NaCl",
                    position=pos_arr.copy(),
                    max_concentration=1.0,
                    decay_constant=FOOD_GRADIENT_DECAY,
                    valence="attractive",
                ),
                ChemSource(
                    molecule="butanone",
                    position=pos_arr.copy(),
                    max_concentration=0.8,
                    decay_constant=FOOD_GRADIENT_DECAY * 0.7,
                    valence="attractive",
                ),
            ]
            self._food_items.append((pos_arr, sources))

        # Non-food sources (aversive odour) — never removed
        self._other_sources: list[ChemSource] = [
            ChemSource(
                molecule="2-nonanone",
                position=np.array([-0.035, 0.02, 0.0]),
                max_concentration=0.6,
                decay_constant=FOOD_GRADIENT_DECAY * 1.5,
                valence="aversive",
            )
        ]

        self._add_nociceptive = add_nociceptive
        self._current_head_pos = np.zeros(3)

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self) -> EnvironmentObservation:
        self._current_head_pos = np.zeros(3)
        return self._build_observation()

    def step(self, body_state: dict[str, Any]) -> EnvironmentObservation:
        """Update observation given the worm's current body state."""
        head_pos = body_state.get("head_position", np.zeros(3))
        if isinstance(head_pos, np.ndarray):
            self._current_head_pos = head_pos.copy()
        else:
            self._current_head_pos = np.array(head_pos)

        # Remove food when worm head touches it
        pos = self._current_head_pos
        self._food_items = [
            (fpos, srcs)
            for fpos, srcs in self._food_items
            if float(np.linalg.norm(pos - fpos)) > FOOD_CONSUMPTION_RADIUS_M
        ]

        return self._build_observation()

    def render(self) -> np.ndarray | None:
        """
        Render a top-down 2D map of chemical gradients as an RGB image.
        Returns (256, 256, 3) uint8 array.
        """
        size = 256
        half = self._plate_radius
        img = np.ones((size, size, 3), dtype=np.uint8) * 220  # light grey background

        xs = np.linspace(-half, half, size)
        ys = np.linspace(-half, half, size)
        xx, yy = np.meshgrid(xs, ys)

        # Render NaCl gradient as green channel (first remaining food)
        nacl_src = None
        for _, srcs in self._food_items:
            for s in srcs:
                if s.molecule == "NaCl":
                    nacl_src = s
                    break
            if nacl_src is not None:
                break
        if nacl_src is not None:
            r = np.sqrt(
                (xx - nacl_src.position[0]) ** 2
                + (yy - nacl_src.position[1]) ** 2
            )
            conc = nacl_src.max_concentration * np.exp(
                -nacl_src.decay_constant * r
            )
            conc_norm = (conc * 255).astype(np.uint8)
            img[:, :, 1] = np.clip(220 - conc_norm * 2, 100, 220).astype(np.uint8)

        # Draw worm head position
        px = int((self._current_head_pos[0] + half) / (2 * half) * size)
        py = int((self._current_head_pos[1] + half) / (2 * half) * size)
        px = np.clip(px, 2, size - 3)
        py = np.clip(py, 2, size - 3)
        img[py - 2 : py + 3, px - 2 : px + 3] = [255, 50, 50]

        return img

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_chemical_concentration(
        self, molecule: str, position: np.ndarray
    ) -> float:
        best = 0.0
        for _, srcs in self._food_items:
            for src in srcs:
                if src.molecule == molecule:
                    best = max(best, src.concentration_at(position))
        for src in self._other_sources:
            if src.molecule == molecule:
                best = max(best, src.concentration_at(position))
        return min(best, CHEM_CONCENTRATION_MAX)

    def is_on_plate(self, position: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(position[:2]) <= self._plate_radius
        )

    def is_nociceptive(self, position: np.ndarray) -> bool:
        if not self._add_nociceptive:
            return False
        return bool(
            np.linalg.norm(position - self._noci_center) <= self._noci_radius
        )

    def get_active_food_positions(self) -> list[tuple[float, float, float]]:
        """Return positions of food sources that have not yet been consumed."""
        return [tuple(p.tolist()) for p, _ in self._food_items]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_observation(self) -> EnvironmentObservation:
        pos = self._current_head_pos

        # Use max (strongest nearby source) per molecule to avoid saturation
        # that destroys the gradient when multiple sources overlap.
        chemicals: dict[str, float] = {}
        for _, srcs in self._food_items:
            for src in srcs:
                conc = src.concentration_at(pos)
                chemicals[src.molecule] = max(
                    chemicals.get(src.molecule, 0.0), conc
                )
        for src in self._other_sources:
            conc = src.concentration_at(pos)
            chemicals[src.molecule] = max(
                chemicals.get(src.molecule, 0.0), conc
            )

        # Nociceptive signal → delivered to ASH neurons via contact_forces
        contact_forces: dict[str, np.ndarray] = {}
        if self.is_nociceptive(pos):
            noci_intensity = 1.0 - float(
                np.linalg.norm(pos - self._noci_center) / self._noci_radius
            )
            contact_forces["nociceptive"] = np.array(
                [float(noci_intensity), 0.0, 0.0]
            )

        # Plate boundary → mechanosensory input (worm hits wall)
        dist_to_edge = self._plate_radius - float(np.linalg.norm(pos[:2]))
        if dist_to_edge < 0.002:  # within 2mm of edge
            wall_force = max(0.0, 1.0 - dist_to_edge / 0.002)
            contact_forces["wall"] = np.array([wall_force, 0.0, 0.0])

        return EnvironmentObservation(
            chemicals=chemicals,
            contact_forces=contact_forces,
            proprioception={},
        )
