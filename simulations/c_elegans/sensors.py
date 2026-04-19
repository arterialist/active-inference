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
    MOTOR_NEURON_POSITIONS,
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

    # B-type motor neuron proprioception (Wen et al. 2012).
    #
    # Real B-motor neurons have processes extending ~2-6 segments
    # *anteriorly* from the neuron soma and sample stretch non-locally
    # along that process. The non-locality is architecturally critical:
    # if DB_N senses joint _at_ segment N, any dorsal bend DB_N produces
    # feeds right back into DB_N as positive feedback and locks the
    # segment in a deep curl (the freeze failure mode documented in
    # tuning/notes.md). Reading a joint 2 segments anterior means DB_N
    # is driven by the bend of segments N-2/N-1 — which DB_N cannot
    # directly amplify — so the wave propagates posteriorly without
    # each segment trapping itself.
    #
    # Offset is 4 segments anterior; clipped at 0 for the most anterior
    # motor neurons so they effectively read the head.  Larger offsets
    # stretch the emergent wavelength along the body: with offset=2 the
    # body carried ~2 wavelengths (fragmented wave, lateral slip > 0.6);
    # offset=4 lengthens the coupling so a single S-wave spans the worm,
    # matching the ~1 wavelength observed in real C. elegans crawling.
    _PROPRIO_ANT_OFFSET: int = 4
    MOTOR_PROPRIO: dict[str, int] = {}
    for _name, _frac in MOTOR_NEURON_POSITIONS.items():
        _prefix = _name.rstrip("0123456789")
        if _prefix in ("DB", "VB"):
            _seg = int(_frac * (N_BODY_SEGMENTS - 1))
            MOTOR_PROPRIO[_name] = max(0, _seg - _PROPRIO_ANT_OFFSET)

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

        # --- Motor proprioception (B-type wave propagation) ---
        inputs.update(self._encode_motor_proprioception(body_state))

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
        """Encode joint curvature as stretch receptor inputs.

        The MJCF body (see ``body_model.xml``) defines each inter-segment
        link as a two-axis hinge: ``j{i}{j}_pitch`` (D/V) and
        ``j{i}{j}_yaw`` (L/R). All 48 body-wall actuators drive **yaw**,
        so yaw is the locomotion plane. Pitch is heavily damped (2.0 vs
        0.05) and sits near zero. Reading pitch here gives no feedback;
        we read yaw so stretch receptors actually see the worm bend.

        This is the ``PROPRIO_MOTOR_GAIN`` path for DVA / PVD interneurons
        — not the B-type motor loop (see ``_encode_motor_proprioception``).
        """
        result: dict[str, float] = {}

        locomotion_angles = [
            angle
            for jname, angle in body_state.joint_angles.items()
            if "yaw" in jname
        ]

        if not locomotion_angles:
            for neuron, _ in self.PROPRIO_NEURONS:
                result[neuron] = 0.0
            return result

        arr = np.array(locomotion_angles)

        for neuron, seg_idx in self.PROPRIO_NEURONS:
            if neuron == "DVA":
                curvature = float(np.mean(np.abs(arr)))
            else:
                idx = min(seg_idx, len(arr) - 1)
                curvature = float(abs(arr[idx]))

            normalised = float(
                np.clip(curvature / JOINT_ANGLE_MAX_RAD, 0.0, 1.0)
            )
            result[neuron] = normalised

        return result

    def _encode_motor_proprioception(
        self, body_state: BodyState
    ) -> dict[str, float]:
        """B-type motor neuron proprioception (Wen et al. 2012 signed).

        Each DB/VB motor neuron senses the *signed* yaw curvature of a
        joint ~2 segments anterior to its soma (see ``MOTOR_PROPRIO``
        for offset choice).

        The stretch-receptor convention comes from Wen et al. 2012 Fig 3:
        when the anterior body bends dorsally (positive yaw in our
        convention), the VENTRAL side is stretched and its stretch
        receptor excites the ventral motor neuron VB, so the posterior
        segment bends ventrally. The chain alternates D/V/D/V segment
        by segment and produces a propagating S-wave.

        Sign convention here:
            VB (ventral excitor): fires *more* when the anterior joint
                is bent dorsally (+yaw).
            DB (dorsal excitor): fires *more* when the anterior joint
                is bent ventrally (−yaw).

        The non-locality (reading ~2 segments anterior) keeps each
        segment from locking onto its own bend; combined with the
        sign alternation above this reproduces the chain-reflex wave
        of Wen 2012.

        Output key is ``_mpr_{neuron_name}`` in ``[-1, 1]``; the caller
        multiplies by ``PROPRIO_MOTOR_GAIN`` and adds to S. The receiver
        clips at 0 if negative to keep firing rates non-negative.
        """
        result: dict[str, float] = {}
        locomotion_angles = [
            angle
            for jname, angle in body_state.joint_angles.items()
            if "yaw" in jname
        ]
        if not locomotion_angles:
            return result
        for neuron_name, joint_idx in self.MOTOR_PROPRIO.items():
            if joint_idx >= len(locomotion_angles):
                continue
            curvature = locomotion_angles[joint_idx]
            prefix = neuron_name.rstrip("0123456789")
            # VB excited by dorsal (+yaw) anterior bend; DB by ventral
            # (−yaw). This is the alternating stretch reflex that
            # builds a traveling wave rather than a whole-body curl.
            sign = -1.0 if prefix == "DB" else 1.0
            result[f"_mpr_{neuron_name}"] = float(
                np.tanh(sign * curvature / JOINT_ANGLE_MAX_RAD)
            )
        return result
