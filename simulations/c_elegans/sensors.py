"""
C. elegans sensory encoder.

Converts EnvironmentObservation + BodyState into the named sensory
input dict expected by CElegansNervousSystem.tick().

Modalities:
  - Chemosensory (ASEL/ASER, AWC, AWB, ASH …)  ← chemical concentrations
  - Mechanosensory (PLM/ALM/AVM/PVM)            ← touch forces
  - Proprioception (stretch receptors DVA, PVDR) ← joint angles/velocity

All signals are normalised to [0, 1] before being injected into PAULA.
"""

from __future__ import annotations

import numpy as np

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.c_elegans.config import (
    CHEMOSENSORY_NEURONS,
    TOUCH_NEURONS,
    CHEM_CONCENTRATION_MAX,
    JOINT_ANGLE_MAX_RAD,
    N_BODY_SEGMENTS,
)


class SensorEncoder:
    """
    Translates multi-modal observations into per-neuron scalar inputs.

    Usage::

        encoder = SensorEncoder()
        inputs = encoder.encode(observation, body_state)
        # inputs: dict[neuron_name -> float in [0,1]]
    """

    # Mapping: chemosensory neuron name -> which chemical molecule it responds to
    # Each neuron is sensitive to one primary ligand.
    CHEM_NEURON_LIGAND: dict[str, str] = {
        "ASEL": "NaCl",
        "ASER": "NaCl",
        "AWCL": "butanone",
        "AWCR": "butanone",
        "AWBL": "2-nonanone",
        "AWBR": "2-nonanone",
        "AFDL": "temperature",
        "AFDR": "temperature",
        "ASHL": "nociceptive",
        "ASHR": "nociceptive",
        "ASJL": "ascaroside",
        "ASJR": "ascaroside",
        "AIZL": "NaCl",     # AIA gets NaCl indirect but for simplicity map to NaCl
        "AIZR": "NaCl",
    }

    # Mapping: touch neuron -> body site it innervates
    TOUCH_NEURON_SITE: dict[str, str] = {
        "PLML": "touch_post_sensor",
        "PLMR": "touch_post_sensor",
        "ALML": "touch_ant_sensor",
        "ALMR": "touch_ant_sensor",
        "AVM":  "touch_ant_sensor",
        "PVM":  "touch_post_sensor",
    }

    # Proprioceptive neurons -> which segment they monitor (0-based index)
    # DVA (interneuron) responds to body curvature across all segments
    # PVDR/PVDL monitor posterior
    PROPRIO_NEURONS: list[tuple[str, int]] = [
        ("PVDL", 10),
        ("PVDR", 10),
        ("DVA",   6),   # DVA integrates mid-body curvature
    ]

    def encode(
        self,
        observation: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        """
        Produce a flat {neuron_name: normalised_intensity} dict.

        Args:
            observation:  Sensory environment observation.
            body_state:   Current physics body state.

        Returns:
            Dict of neuron name -> input intensity in [0, 1].
        """
        inputs: dict[str, float] = {}

        # --- Chemosensory ---
        inputs.update(self._encode_chemical(observation))

        # --- Mechanosensory ---
        inputs.update(self._encode_touch(observation, body_state))

        # --- Proprioception ---
        inputs.update(self._encode_proprioception(body_state))

        return inputs

    # ------------------------------------------------------------------
    # Private encoding methods
    # ------------------------------------------------------------------

    def _encode_chemical(
        self, obs: EnvironmentObservation
    ) -> dict[str, float]:
        """Map chemical concentrations to chemosensory neuron inputs."""
        result: dict[str, float] = {}
        for neuron, ligand in self.CHEM_NEURON_LIGAND.items():
            conc = obs.chemicals.get(ligand, 0.0)
            normalised = float(np.clip(conc / CHEM_CONCENTRATION_MAX, 0.0, 1.0))
            result[neuron] = normalised
        return result

    def _encode_touch(
        self,
        obs: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        """Map contact forces at body sites to mechanosensory neuron inputs."""
        result: dict[str, float] = {}
        for neuron, site in self.TOUCH_NEURON_SITE.items():
            # Force from environment observation (richer) OR from body state
            force_vec = obs.contact_forces.get(
                site,
                body_state.contact_forces.get(site, np.zeros(3)),
            )
            magnitude = float(np.linalg.norm(force_vec))
            # Saturating sigmoid-like normalisation: half-max at 1 nN (biological scale)
            normalised = float(np.clip(magnitude / (magnitude + 1e-9), 0.0, 1.0))
            result[neuron] = normalised
        return result

    def _encode_proprioception(
        self, body_state: BodyState
    ) -> dict[str, float]:
        """Encode joint curvature as stretch receptor inputs."""
        result: dict[str, float] = {}

        # Build a list of pitch angles (dorsal/ventral bending) for all joints
        pitch_angles = [
            angle
            for jname, angle in body_state.joint_angles.items()
            if "pitch" in jname
        ]

        if not pitch_angles:
            for neuron, _ in self.PROPRIO_NEURONS:
                result[neuron] = 0.0
            return result

        pitch_arr = np.array(pitch_angles)

        for neuron, seg_idx in self.PROPRIO_NEURONS:
            if neuron == "DVA":
                # DVA integrates total body curvature
                curvature = float(np.mean(np.abs(pitch_arr)))
            else:
                # PLM/PVD-like: local angle at the relevant segment
                idx = min(seg_idx, len(pitch_arr) - 1)
                curvature = float(abs(pitch_arr[idx]))

            normalised = float(
                np.clip(curvature / JOINT_ANGLE_MAX_RAD, 0.0, 1.0)
            )
            result[neuron] = normalised

        return result
