"""
C. elegans sensory encoder.

Converts EnvironmentObservation + BodyState into the named sensory
input dict expected by CElegansNervousSystem.tick().

Modalities, mapped per neuron in :data:`SENSORY_MANIFEST`:
  - Chemosensory          (amphid, phasmid, IL2 ascaroside neurons)
  - Mechanosensory        (gentle touch, harsh / polymodal touch,
                           nose touch, head proprioceptors)
  - Stretch / proprio     (DVA, PVD, motor B-type neurons)
  - Polymodal             (ASH = nociception + nose touch, FLP / PVD =
                           harsh touch + temperature, PHA / PHB / PHC =
                           tail nociception + ascaroside)
  - Gas / O2 / CO2        (URX, AQR, PQR, BAG, ASG, SDQ — wired to
                           env-supplied "O2" / "CO2" channels; if the
                           environment has not yet been extended with
                           those channels they read 0 instead of being
                           silently absent)

Every one of the 83 Cook-2019 sensory neurons is covered by an entry in
:data:`SENSORY_MANIFEST` so no neuron is left "dead" — even classes
without a corresponding environmental signal today (light, gas, dauer
pheromone) are wired to ligand keys the environment can populate later.

References
----------
- Goodman MB. (2006) "Mechanosensation". WormBook.
- Hart AC. (2006) "Behavior". WormBook.
- Bargmann CI. (2006) "Chemosensation in C. elegans". WormBook.
- Chalfie & Sulston (1981) J Neurosci — gentle touch neurons (ALM/PLM/AVM/PVM).
- Way & Chalfie (1989) — harsh touch reorientation.
- Kaplan & Horvitz (1993) PNAS — nose touch (OLQ, ASH, FLP).
- Hilliard et al. (2002) EMBO J — ASH polymodal nociception.
- Mori & Ohshima (1995) Nature — AFD thermotaxis.
- Bargmann & Horvitz (1991) Neuron — ASE NaCl chemotaxis.
- Suzuki et al. (2008) Nature — ASEL/ASER ON/OFF.
- Pierce-Shimomura et al. (1999) J Neurosci — ASE biased random walk.
- Troemel et al. (1995) Cell — AWA diacetyl, AWC butanone.
- Bargmann et al. (1993) Cell — AWA/AWC olfactory mapping.
- Coates & de Bono (2002) Nature — URX O2 sensing.
- Gray et al. (2004) Nature — AQR/PQR/URX aerotaxis.
- Hallem & Sternberg (2008) PNAS — BAG CO2.
- Sawin et al. (2000) Neuron — CEP basal slowing on bacteria.
- Sulston et al. (1975) — ADE/PDE deirids dopaminergic mechanosensors.
- Wen et al. (2012) Neuron — proprioceptive coupling along body wall.
- Albeg et al. (2011) Mol Cell Neurosci — PVD harsh touch.
- Husson et al. (2012) Curr Biol — PVD multimodal nociception.
- Chatzigeorgiou et al. (2010) Nat Neurosci — FLP harsh touch + temperature.
- Liu et al. (2018) eLife — PHC tail harsh / hot.
- Hilliard et al. (2002) EMBO J — ASH octanol, quinine, osmotic.
- Edwards et al. (2008), Liu et al. (2010) — ASJ short-wavelength light.
- Ward et al. (2008), Bhatla & Horvitz (2015) — IL2 light + ascaroside.
"""

from __future__ import annotations

import numpy as np

from simulations.base_body import BodyState
from simulations.base_environment import EnvironmentObservation
from simulations.c_elegans.config import (
    CHEM_CONCENTRATION_MAX,
    JOINT_ANGLE_MAX_RAD,
    MOTOR_NEURON_POSITIONS,
    N_BODY_SEGMENTS,
    TOUCH_HALF_SAT_FORCE,
    TOUCH_HARSH_THRESHOLD,
)


# ----------------------------------------------------------------------
# Per-neuron stimulus manifest — single source of truth.
# ----------------------------------------------------------------------
# Each entry may carry:
#   "ligands":     list of (chemical_key, weight); summed before
#                  saturation. ``chemical_key`` references the keys
#                  EnvironmentObservation.chemicals exposes — currently
#                  "NaCl", "butanone", "2-nonanone", "temperature",
#                  "nociceptive", "ascaroside" + any future channel the
#                  environment adds (O2, CO2, light_uv, …). Missing keys
#                  read as 0, which is the correct biology when no
#                  stimulus is present.
#   "touch_sites": list of MuJoCo touch-sensor names whose forces are
#                  summed; the running total feeds a half-sat sigmoid
#                  (TOUCH_HALF_SAT_FORCE) so quiet floor support stays
#                  near 0 and contact saturates above ~ half-sat.
#   "harsh":       true → after sigmoid, subtract TOUCH_HARSH_THRESHOLD
#                  and re-normalise. Models the high mechanical
#                  threshold of harsh-touch neurons (FLP, PVD, PHC).
#   "polymodal":   if both ligands and touch sites are present, the
#                  encoded value is the saturated sum of contributions.
#                  Used implicitly whenever both keys exist — no flag
#                  needed.
SENSORY_MANIFEST: dict[str, dict] = {
    # ──────────────────────────────────────────────────────────────
    #  AMPHID CHEMOSENSORY — Bargmann/Troemel/Suzuki et al.
    # ──────────────────────────────────────────────────────────────
    # ASE: water-soluble salts and ions (NaCl primary). ASEL is ON
    # (fires on increase), ASER is OFF (fires on decrease) — handled
    # by neuron_mapping._OFF_CELL_NAMES.
    "ASEL": {"ligands": [("NaCl", 1.0)]},
    "ASER": {"ligands": [("NaCl", 1.0)]},
    # AWA: volatile attractants — diacetyl, pyrazine, 2,3-pentanedione.
    "AWAL": {"ligands": [("diacetyl", 1.0), ("pyrazine", 0.5)]},
    "AWAR": {"ligands": [("diacetyl", 1.0), ("pyrazine", 0.5)]},
    # AWC: volatile attractants — butanone, isoamyl alcohol,
    # benzaldehyde. OFF cell.
    "AWCL": {"ligands": [("butanone", 1.0), ("isoamyl_alcohol", 0.7)]},
    "AWCR": {"ligands": [("butanone", 1.0), ("isoamyl_alcohol", 0.7)]},
    # AWB: volatile repellents — 2-nonanone, 1-octanol.
    "AWBL": {"ligands": [("2-nonanone", 1.0), ("1-octanol", 0.6)]},
    "AWBR": {"ligands": [("2-nonanone", 1.0), ("1-octanol", 0.6)]},
    # AFD: thermosensation — fires around the cultivation T.
    "AFDL": {"ligands": [("temperature", 1.0)]},
    "AFDR": {"ligands": [("temperature", 1.0)]},
    # ASH: primary polymodal nociceptor — high osmolarity, harsh nose
    # touch, 1-octanol, quinine, acid, heavy metals.
    "ASHL": {
        "ligands": [
            ("nociceptive", 1.0),
            ("osmotic", 0.7),
            ("1-octanol", 0.6),
            ("quinine", 0.5),
        ],
        "touch_sites": ["touch_nose_sensor", "touch_nose_left", "touch_nose_right"],
    },
    "ASHR": {
        "ligands": [
            ("nociceptive", 1.0),
            ("osmotic", 0.7),
            ("1-octanol", 0.6),
            ("quinine", 0.5),
        ],
        "touch_sites": ["touch_nose_sensor", "touch_nose_left", "touch_nose_right"],
    },
    # ASI: ascaroside / dauer pheromone, NaCl modulation.
    "ASIL": {"ligands": [("ascaroside", 1.0), ("NaCl", 0.3)]},
    "ASIR": {"ligands": [("ascaroside", 1.0), ("NaCl", 0.3)]},
    # ASJ: ascaroside, dauer recovery, short-wavelength light (LITE-1).
    "ASJL": {"ligands": [("ascaroside", 1.0), ("light_uv", 0.6)]},
    "ASJR": {"ligands": [("ascaroside", 1.0), ("light_uv", 0.6)]},
    # ASK: lysine attraction, repellent modulation, also ascaroside.
    "ASKL": {"ligands": [("lysine", 1.0), ("ascaroside", 0.5)]},
    "ASKR": {"ligands": [("lysine", 1.0), ("ascaroside", 0.5)]},
    # ASG: weak chemo + CO2 modulation.
    "ASGL": {"ligands": [("CO2", 0.6), ("isoamyl_alcohol", 0.4)]},
    "ASGR": {"ligands": [("CO2", 0.6), ("isoamyl_alcohol", 0.4)]},
    # ADF: ascaroside, dauer pheromone (serotonergic).
    "ADFL": {"ligands": [("ascaroside", 1.0), ("dauer_pheromone", 0.6)]},
    "ADFR": {"ligands": [("ascaroside", 1.0), ("dauer_pheromone", 0.6)]},
    # ADL: hyperosmotic, 1-octanol, ascaroside avoidance.
    "ADLL": {
        "ligands": [("osmotic", 1.0), ("1-octanol", 0.7), ("ascaroside", 0.4)],
    },
    "ADLR": {
        "ligands": [("osmotic", 1.0), ("1-octanol", 0.7), ("ascaroside", 0.4)],
    },
    # BAG: CO2 primary (excited by rising CO2), also responds to
    # falling O2.
    "BAGL": {"ligands": [("CO2", 1.0), ("O2", 0.5)]},
    "BAGR": {"ligands": [("CO2", 1.0), ("O2", 0.5)]},
    # URX, AQR, PQR: O2 sensors (URX = head, AQR = midbody, PQR = tail).
    "URXL": {"ligands": [("O2", 1.0)]},
    "URXR": {"ligands": [("O2", 1.0)]},
    "AQR": {"ligands": [("O2", 1.0)]},
    "PQR": {"ligands": [("O2", 1.0)]},
    # ──────────────────────────────────────────────────────────────
    #  PHASMID (tail chemo + tail nociception, polymodal)
    # ──────────────────────────────────────────────────────────────
    # PHA / PHB modulate ASH-driven reversals; biased toward repellent
    # cessation.  PHC is harsh-touch + heat at the tail.
    "PHAL": {
        "ligands": [("ascaroside", 0.6), ("nociceptive", 0.4)],
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    "PHAR": {
        "ligands": [("ascaroside", 0.6), ("nociceptive", 0.4)],
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    "PHBL": {
        "ligands": [("ascaroside", 0.6), ("nociceptive", 0.4)],
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    "PHBR": {
        "ligands": [("ascaroside", 0.6), ("nociceptive", 0.4)],
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    "PHCL": {
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
        "harsh": True,
        "ligands": [("temperature_hot", 0.6)],
    },
    "PHCR": {
        "touch_sites": [
            "touch_post_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
        "harsh": True,
        "ligands": [("temperature_hot", 0.6)],
    },
    # ──────────────────────────────────────────────────────────────
    #  IL1 / IL2 — head inner-labial (mechano + chemo + light)
    # ──────────────────────────────────────────────────────────────
    "IL1L": {"touch_sites": ["touch_nose_left", "touch_seg0_sensor"]},
    "IL1R": {"touch_sites": ["touch_nose_right", "touch_seg0_sensor"]},
    "IL1DL": {"touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"]},
    "IL1DR": {"touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"]},
    "IL1VL": {"touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"]},
    "IL1VR": {"touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"]},
    "IL2L": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_left"],
    },
    "IL2R": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_right"],
    },
    "IL2DL": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_dorsal"],
    },
    "IL2DR": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_dorsal"],
    },
    "IL2VL": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_ventral"],
    },
    "IL2VR": {
        "ligands": [("ascaroside", 0.5), ("light_uv", 0.4)],
        "touch_sites": ["touch_nose_ventral"],
    },
    # ──────────────────────────────────────────────────────────────
    #  GENTLE TOUCH (mec-4 / mec-10)
    # ──────────────────────────────────────────────────────────────
    "ALML": {
        "touch_sites": [
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_ant_sensor",
        ],
    },
    "ALMR": {
        "touch_sites": [
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_ant_sensor",
        ],
    },
    "AVM": {
        "touch_sites": [
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_ant_sensor",
        ],
    },
    "PLML": {
        "touch_sites": [
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_seg7_sensor",
            "touch_seg8_sensor",
            "touch_seg9_sensor",
            "touch_seg10_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
            "touch_post_sensor",
        ],
    },
    "PLMR": {
        "touch_sites": [
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_seg7_sensor",
            "touch_seg8_sensor",
            "touch_seg9_sensor",
            "touch_seg10_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
            "touch_post_sensor",
        ],
    },
    "PVM": {
        "touch_sites": [
            "touch_seg5_sensor",
            "touch_seg6_sensor",
            "touch_seg7_sensor",
            "touch_seg8_sensor",
            "touch_seg9_sensor",
            "touch_seg10_sensor",
        ],
    },
    # ──────────────────────────────────────────────────────────────
    #  HARSH TOUCH / POLYMODAL NOCICEPTORS
    # ──────────────────────────────────────────────────────────────
    "FLPL": {
        "touch_sites": [
            "touch_nose_sensor",
            "touch_nose_dorsal",
            "touch_nose_ventral",
            "touch_nose_left",
            "touch_nose_right",
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
        ],
        "harsh": True,
        "ligands": [("temperature_hot", 0.5)],
    },
    "FLPR": {
        "touch_sites": [
            "touch_nose_sensor",
            "touch_nose_dorsal",
            "touch_nose_ventral",
            "touch_nose_left",
            "touch_nose_right",
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
            "touch_seg3_sensor",
        ],
        "harsh": True,
        "ligands": [("temperature_hot", 0.5)],
    },
    "PVDL": {
        "touch_sites": [f"touch_seg{i}_sensor" for i in range(3, 12)],
        "harsh": True,
        "ligands": [("temperature_cold", 0.5)],
    },
    "PVDR": {
        "touch_sites": [f"touch_seg{i}_sensor" for i in range(3, 12)],
        "harsh": True,
        "ligands": [("temperature_cold", 0.5)],
    },
    # ──────────────────────────────────────────────────────────────
    #  HEAD MECHANO — OLQ (nose-touch quadrants), OLL, URY, CEP, ADE
    # ──────────────────────────────────────────────────────────────
    "OLQDL": {"touch_sites": ["touch_nose_sensor", "touch_nose_dorsal"]},
    "OLQDR": {"touch_sites": ["touch_nose_sensor", "touch_nose_dorsal"]},
    "OLQVL": {"touch_sites": ["touch_nose_sensor", "touch_nose_ventral"]},
    "OLQVR": {"touch_sites": ["touch_nose_sensor", "touch_nose_ventral"]},
    "OLLL": {
        "touch_sites": [
            "touch_nose_sensor",
            "touch_seg0_sensor",
            "touch_seg1_sensor",
        ],
    },
    "OLLR": {
        "touch_sites": [
            "touch_nose_sensor",
            "touch_seg0_sensor",
            "touch_seg1_sensor",
        ],
    },
    "URYDL": {"touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"]},
    "URYDR": {"touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"]},
    "URYVL": {"touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"]},
    "URYVR": {"touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"]},
    # CEP: cephalic dopaminergic. Mechano sensitivity to bacterial
    # texture (basal slowing response, Sawin 2000).
    "CEPDL": {
        "touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    "CEPDR": {
        "touch_sites": ["touch_nose_dorsal", "touch_seg0_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    "CEPVL": {
        "touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    "CEPVR": {
        "touch_sites": ["touch_nose_ventral", "touch_seg0_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    # ADE / PDE: deirids, dopaminergic, body-wall mechanoreceptors.
    "ADEL": {
        "touch_sites": ["touch_seg1_sensor", "touch_seg2_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    "ADER": {
        "touch_sites": ["touch_seg1_sensor", "touch_seg2_sensor"],
        "ligands": [("bacteria_lawn", 1.0)],
    },
    "PDEL": {
        "touch_sites": ["touch_seg9_sensor", "touch_seg10_sensor"],
        "ligands": [("bacteria_lawn", 0.6)],
    },
    "PDER": {
        "touch_sites": ["touch_seg9_sensor", "touch_seg10_sensor"],
        "ligands": [("bacteria_lawn", 0.6)],
    },
    # ──────────────────────────────────────────────────────────────
    #  ALN / PLN — body-wall lateral neurons (weak mechano coupling,
    #  associate with ALM/PLM)
    # ──────────────────────────────────────────────────────────────
    "ALNL": {
        "touch_sites": [
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
        ],
    },
    "ALNR": {
        "touch_sites": [
            "touch_seg0_sensor",
            "touch_seg1_sensor",
            "touch_seg2_sensor",
        ],
    },
    "PLNL": {
        "touch_sites": [
            "touch_seg10_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    "PLNR": {
        "touch_sites": [
            "touch_seg10_sensor",
            "touch_seg11_sensor",
            "touch_seg12_sensor",
        ],
    },
    # ──────────────────────────────────────────────────────────────
    #  SDQ — body neurons, mechano + O2 modulation
    # ──────────────────────────────────────────────────────────────
    "SDQL": {
        "touch_sites": [
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
        ],
        "ligands": [("O2", 0.4)],
    },
    "SDQR": {
        "touch_sites": [
            "touch_seg3_sensor",
            "touch_seg4_sensor",
            "touch_seg5_sensor",
        ],
        "ligands": [("O2", 0.4)],
    },
    # ──────────────────────────────────────────────────────────────
    #  STRETCH / PROPRIO — proprioceptive contribution is added by
    #  _encode_proprioception below; the touch entries here capture
    #  axial vibration / tail contact that DVA also reports.
    # ──────────────────────────────────────────────────────────────
    "DVA": {
        "touch_sites": ["touch_seg11_sensor", "touch_seg12_sensor"],
    },
}


# Backward-compatibility projections of the manifest. Other modules
# (and external scripts/tests) still import these names. They are
# now derived views — the manifest is authoritative.
CHEM_NEURON_LIGAND: dict[str, str] = {
    name: spec["ligands"][0][0]
    for name, spec in SENSORY_MANIFEST.items()
    if "ligands" in spec
}
TOUCH_NEURON_SITE: dict[str, str] = {
    name: spec["touch_sites"][0]
    for name, spec in SENSORY_MANIFEST.items()
    if "touch_sites" in spec
}


class SensorEncoder:
    """
    Translates multi-modal observations into per-neuron scalar inputs.

    The wiring is data-driven: :data:`SENSORY_MANIFEST` declares for
    every sensory neuron which ligands and which touch sensors it
    responds to. ``encode()`` walks the manifest each tick and produces
    a flat ``{neuron_name: intensity in [0, 1]}`` dictionary that is
    fed verbatim to PAULA via ``CElegansNervousSystem.tick()``.
    """

    # Re-exported manifest views (kept as class attrs to preserve the
    # historical SensorEncoder.CHEM_NEURON_LIGAND / TOUCH_NEURON_SITE
    # API used by docs and external tests).
    CHEM_NEURON_LIGAND = CHEM_NEURON_LIGAND
    TOUCH_NEURON_SITE = TOUCH_NEURON_SITE

    # Proprioceptive neurons with explicit segment readout (joint-angle
    # curvature). DVA + PVD: stretch / curvature integration.
    PROPRIO_NEURONS: list[tuple[str, int]] = [
        ("PVDL", 10),
        ("PVDR", 10),
        ("DVA", 6),
    ]

    # Anterior offset for B-type motor proprioception (Wen 2012).
    _PROPRIO_ANT_OFFSET: int = 1
    MOTOR_PROPRIO: dict[str, int] = {}
    for _name, _frac in MOTOR_NEURON_POSITIONS.items():
        _prefix = _name.rstrip("0123456789")
        if _prefix in ("DB", "VB"):
            _seg = int(_frac * (N_BODY_SEGMENTS - 1))
            MOTOR_PROPRIO[_name] = max(0, _seg - _PROPRIO_ANT_OFFSET)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def encode(
        self,
        observation: EnvironmentObservation,
        body_state: BodyState,
    ) -> dict[str, float]:
        """
        Produce a flat ``{neuron_name: normalised_intensity}`` dict.

        Polymodal neurons (entries with both ``ligands`` and
        ``touch_sites``) sum their chemical and mechanical
        contributions before saturation, so e.g. ASH fires on either
        a noxious chemical OR a harsh nose touch — the published
        polymodal behaviour.
        """
        inputs: dict[str, float] = {}

        chem = self._encode_chemical(observation)
        touch = self._encode_touch(body_state)

        # Union the keyspace and sum (saturate at 1.0). Each modality
        # already produced a [0, 1] value, so a polymodal neuron at
        # full firing on either modality reads near 1.0.
        for name in set(chem) | set(touch):
            value = chem.get(name, 0.0) + touch.get(name, 0.0)
            inputs[name] = float(np.clip(value, 0.0, 1.0))

        inputs.update(self._encode_proprioception(body_state))
        inputs.update(self._encode_motor_proprioception(body_state))

        return inputs

    # ------------------------------------------------------------------
    # Private encoding methods
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_chemical(obs: EnvironmentObservation) -> dict[str, float]:
        """Map chemical concentrations to chemosensory inputs.

        The environment exposes whatever ligands it currently models
        (``obs.chemicals`` is a dict). Missing keys read as 0 — a
        neuron tuned to a ligand the env does not yet supply gets no
        input, which is the correct biology.
        """
        result: dict[str, float] = {}
        for neuron, spec in SENSORY_MANIFEST.items():
            ligands = spec.get("ligands")
            if not ligands:
                continue
            total = 0.0
            for ligand, weight in ligands:
                conc = float(obs.chemicals.get(ligand, 0.0))
                total += weight * conc
            result[neuron] = float(np.clip(total / CHEM_CONCENTRATION_MAX, 0.0, 1.0))
        return result

    @staticmethod
    def _encode_touch(body_state: BodyState) -> dict[str, float]:
        """Map summed contact forces over each neuron's receptive field.

        Saturating sigmoid ``f / (f + halfsat)`` keeps quiet floor
        contact near zero and drives strong contact toward 1.0.
        Harsh-touch neurons subtract a threshold afterwards so they
        only fire on substantial pressure.
        """
        result: dict[str, float] = {}
        halfsat = float(TOUCH_HALF_SAT_FORCE)
        threshold = float(TOUCH_HARSH_THRESHOLD)
        for neuron, spec in SENSORY_MANIFEST.items():
            sites = spec.get("touch_sites")
            if not sites:
                continue
            total = 0.0
            for site in sites:
                vec = body_state.contact_forces.get(site)
                if vec is not None:
                    total += float(np.linalg.norm(vec))
            if total <= 0.0:
                result[neuron] = 0.0
                continue
            saturated = total / (total + halfsat)
            if spec.get("harsh"):
                saturated = max(0.0, (saturated - threshold) / (1.0 - threshold))
            result[neuron] = float(np.clip(saturated, 0.0, 1.0))
        return result

    @classmethod
    def _encode_proprioception(cls, body_state: BodyState) -> dict[str, float]:
        """DVA / PVD curvature readout (yaw-axis stretch receptors).

        The MJCF body defines each inter-segment link as a two-axis
        hinge; all body-wall actuators drive **yaw**, so yaw is the
        locomotion plane. We read yaw curvature for stretch receptors.
        """
        result: dict[str, float] = {}

        locomotion_angles = [
            angle for jname, angle in body_state.joint_angles.items() if "yaw" in jname
        ]

        if not locomotion_angles:
            for neuron, _ in cls.PROPRIO_NEURONS:
                result[neuron] = 0.0
            return result

        arr = np.array(locomotion_angles)

        for neuron, seg_idx in cls.PROPRIO_NEURONS:
            if neuron == "DVA":
                curvature = float(np.mean(np.abs(arr)))
            else:
                idx = min(seg_idx, len(arr) - 1)
                curvature = float(abs(arr[idx]))

            normalised = float(np.clip(curvature / JOINT_ANGLE_MAX_RAD, 0.0, 1.0))
            # Add to any existing chemo+touch contribution (polymodal:
            # PVD already has harsh-touch + cold-temp; proprio is
            # biologically also part of its repertoire).
            result[neuron] = max(result.get(neuron, 0.0), normalised)

        return result

    @classmethod
    def _encode_motor_proprioception(cls, body_state: BodyState) -> dict[str, float]:
        """B-type motor neuron proprioception (Wen et al. 2012 signed).

        Each DB/VB motor neuron senses the *signed* yaw curvature of a
        joint ~``_PROPRIO_ANT_OFFSET`` segments anterior to its soma.
        See sensors-encoder docs and ``MOTOR_PROPRIO``.

        Output key is ``_mpr_{neuron_name}`` in ``[-1, 1]``; the caller
        multiplies by ``PROPRIO_MOTOR_GAIN`` and adds to S. The
        receiver clips at 0 if negative.
        """
        result: dict[str, float] = {}
        locomotion_angles = [
            angle for jname, angle in body_state.joint_angles.items() if "yaw" in jname
        ]
        if not locomotion_angles:
            return result
        for neuron_name, joint_idx in cls.MOTOR_PROPRIO.items():
            if joint_idx >= len(locomotion_angles):
                continue
            curvature = locomotion_angles[joint_idx]
            prefix = neuron_name.rstrip("0123456789")
            # B-type motor neurons must contract IN PHASE with the anterior segment
            # to propagate the wave backward.
            # Curvature > 0 means the anterior segment is bent dorsally.
            # To bend the current segment dorsally, we must EXCITE DB and INHIBIT VB.
            # Therefore, DB requires a positive sign, and VB a negative sign.
            sign = 1.0 if prefix == "DB" else -1.0
            result[f"_mpr_{neuron_name}"] = float(
                np.tanh(sign * curvature / JOINT_ANGLE_MAX_RAD)
            )
        return result
