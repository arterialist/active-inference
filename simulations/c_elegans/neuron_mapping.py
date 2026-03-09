"""
C. elegans neuron mapping: connectome -> PAULA NeuronNetwork.

This module bridges the C. elegans-specific connectome data and the
generic PAULA build pipeline, adding:

  1. Biologically appropriate NeuronParameters per neuron class.
  2. A NervousSystem implementation (wraps NeuronNetwork) that knows
     which PAULA neurons receive sensory inputs and which drive muscles.
  3. Sensory encoding and motor decoding at the PAULA tick level.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from simulations.base_nervous_system import BaseNervousSystem
from simulations.connectome_loader import ConnectomeData, build_paula_network
from simulations.c_elegans.config import (
    CHEMOSENSORY_NEURONS,
    TOUCH_NEURONS,
    VENTRAL_CORD_MOTOR_NEURONS,
    LOCOMOTION_INTERNEURONS,
    NEURAL_TICKS_PER_PHYSICS_STEP,
    SPIKE_RATE_TO_ACTIVATION,
    MUSCLE_FILTER_ALPHA,
    MUSCLE_MAX_TORQUE_NM,
    N_BODY_SEGMENTS,
    MUSCLE_QUADRANTS,
)

_PAULA_PATH = Path(__file__).resolve().parents[4] / "neuron-model"
if str(_PAULA_PATH) not in sys.path:
    sys.path.insert(0, str(_PAULA_PATH))

from neuron.neuron import NeuronParameters  # noqa: E402
from neuron.network import NeuronNetwork    # noqa: E402


class CElegansNervousSystem(BaseNervousSystem):
    """
    Complete C. elegans nervous system backed by 302 PAULA neurons.

    Sensory encoding
    ----------------
    Scalar sensory signals (chemicals, touch, proprioception) are written
    directly into the PAULA input_buffer of the corresponding sensory
    neurons via NeuronNetwork.set_external_input().

    Motor decoding
    --------------
    The averaged firing signal of the ventral-cord motor neurons is
    mapped to per-segment muscle activations.  B-type motor neurons
    (DB, VB) are excitatory for forward locomotion; D-type (DD, VD)
    are inhibitory (here modelled as reducing opposing muscle force).
    AS-type neurons contribute to dorsal muscle excitation.
    """

    def __init__(self, connectome: ConnectomeData, log_level: str = "WARNING"):
        self._connectome = connectome
        self._log_level = log_level
        self._network: NeuronNetwork | None = None
        self._name_to_id: dict[str, int] = {}

        # Per-muscle activation state (low-pass filtered)
        self._muscle_activations: dict[str, float] = {}
        self._init_muscles()

        # Build the PAULA network
        self._build()

    # ------------------------------------------------------------------
    # BaseNervousSystem interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._build()
        self._init_muscles()

    def tick(
        self,
        sensory_inputs: dict[str, float],
        current_tick: int,
    ) -> dict[str, float]:
        """
        One PAULA tick.

        Args:
            sensory_inputs: {neuron_name: normalised_intensity, ...}
            current_tick:   Global tick counter.

        Returns:
            Muscle activations: {muscle_name: activation in [0,1], ...}
        """
        if self._network is None:
            raise RuntimeError("Network not initialised — call reset() first")

        # --- inject sensory inputs ---
        self._inject_sensory(sensory_inputs, current_tick)

        # --- run one tick ---
        self._network.run_tick()

        # --- decode motor outputs ---
        return self._decode_motor_outputs()

    def get_neuron_states(self) -> dict[str, Any]:
        """Return membrane potential (S) and last-fired flag for all neurons."""
        if self._network is None:
            return {}
        states = {}
        for nid, neuron in self._network.network.neurons.items():
            name = neuron.metadata.get("name", str(nid))
            states[f"{name}_S"] = neuron.S
            states[f"{name}_fired"] = float(neuron.O > 0)
        return states

    @property
    def n_neurons(self) -> int:
        return len(self._name_to_id)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def name_to_id(self) -> dict[str, int]:
        return dict(self._name_to_id)

    def get_neuron_by_name(self, name: str):
        """Return the PAULA Neuron object for a biological neuron name."""
        nid = self._name_to_id.get(name)
        if nid is None or self._network is None:
            return None
        return self._network.network.neurons.get(nid)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """(Re)build the PAULA NeuronNetwork from the connectome."""
        network, name_to_id = build_paula_network(
            self._connectome,
            base_params=_base_params(),
            sensory_params=_sensory_params(),
            motor_params=_motor_params(),
            interneuron_params=_interneuron_params(),
            # weight_scale: raw connectome synapse count -> PAULA u_i.info
            # 0.3 gives VB/DB class neurons enough drive to fire when a few
            # upstream interneurons are active
            weight_scale=0.3,
            log_level=self._log_level,
        )
        self._network = network
        self._name_to_id = name_to_id
        logger.info(
            f"CElegansNervousSystem ready: "
            f"{len(name_to_id)} neurons in PAULA network"
        )

    def _init_muscles(self) -> None:
        """Initialise per-muscle activation state to zero."""
        self._muscle_activations = {}
        for seg in range(N_BODY_SEGMENTS):
            for quad in MUSCLE_QUADRANTS:
                self._muscle_activations[f"seg{seg}_{quad}"] = 0.0

    def _inject_sensory(
        self, sensory_inputs: dict[str, float], current_tick: int
    ) -> None:
        """Write sensory signals into the network via set_external_input."""
        for neuron_name, intensity in sensory_inputs.items():
            nid = self._name_to_id.get(neuron_name)
            if nid is None:
                continue
            neuron = self._network.network.neurons.get(nid)
            if neuron is None or neuron.params.num_inputs == 0:
                continue
            # Use synapse slot 0 as the "sensory receptor" input slot
            synapse_id = 0
            if synapse_id < neuron.params.num_inputs:
                clamped = float(np.clip(intensity, 0.0, 2.0))
                self._network.set_external_input(
                    neuron_id=nid,
                    synapse_id=synapse_id,
                    info=clamped,
                    mod=np.array([clamped * 0.1, 0.0]),
                )

    def _decode_motor_outputs(self) -> dict[str, float]:
        """
        Convert motor neuron membrane potentials to muscle activations.

        Motor neurons are arranged along the ventral cord with a roughly
        segment-by-segment spatial map.  We use the ordered lists of
        DB/VB (forward excitatory) and DD/VD (inhibitory) neurons and
        map them to the dorsal/ventral muscle quadrants of each segment.
        """
        db_neurons = [f"DB{i}" for i in range(1, 8)]    # 7 dorsal B-type
        vb_neurons = [f"VB{i}" for i in range(1, 12)]   # 11 ventral B-type
        dd_neurons = [f"DD{i}" for i in range(1, 7)]    # 6 dorsal D-type
        vd_neurons = [f"VD{i}" for i in range(1, 14)]   # 13 ventral D-type
        as_neurons = [f"AS{i}" for i in range(1, 12)]   # 11 A-type

        def _firing(name: str) -> float:
            """Return normalised output signal (0 or 1) for a neuron."""
            n = self.get_neuron_by_name(name)
            return float(n.O > 0) if n is not None else 0.0

        # Build per-segment dorsal and ventral excitation
        dorsal_excit = np.zeros(N_BODY_SEGMENTS)
        ventral_excit = np.zeros(N_BODY_SEGMENTS)

        for i, seg in enumerate(range(N_BODY_SEGMENTS)):
            # Map DB neurons (dorsal excitatory) to dorsal muscle
            db_idx = int(i * len(db_neurons) / N_BODY_SEGMENTS)
            db_idx = min(db_idx, len(db_neurons) - 1)
            dorsal_excit[seg] += _firing(db_neurons[db_idx])

            # AS neurons also excite dorsal muscles
            as_idx = int(i * len(as_neurons) / N_BODY_SEGMENTS)
            as_idx = min(as_idx, len(as_neurons) - 1)
            dorsal_excit[seg] += _firing(as_neurons[as_idx]) * 0.5

            # Map VB neurons (ventral excitatory)
            vb_idx = int(i * len(vb_neurons) / N_BODY_SEGMENTS)
            vb_idx = min(vb_idx, len(vb_neurons) - 1)
            ventral_excit[seg] += _firing(vb_neurons[vb_idx])

            # DD/VD neurons provide cross-inhibition (dorsal->ventral,
            # ventral->dorsal) via their inhibitory connections in the
            # connectome.  In this simplified decode we reduce the
            # opposing muscle's signal.
            dd_idx = int(i * len(dd_neurons) / N_BODY_SEGMENTS)
            dd_idx = min(dd_idx, len(dd_neurons) - 1)
            ventral_excit[seg] -= _firing(dd_neurons[dd_idx]) * 0.5

            vd_idx = int(i * len(vd_neurons) / N_BODY_SEGMENTS)
            vd_idx = min(vd_idx, len(vd_neurons) - 1)
            dorsal_excit[seg] -= _firing(vd_neurons[vd_idx]) * 0.5

        dorsal_excit = np.clip(dorsal_excit, 0.0, 1.0)
        ventral_excit = np.clip(ventral_excit, 0.0, 1.0)

        # Apply low-pass filter and build output dict
        for seg in range(N_BODY_SEGMENTS):
            for quad in MUSCLE_QUADRANTS:
                key = f"seg{seg}_{quad}"
                if "D" in quad:   # DL or DR -> dorsal
                    target = float(dorsal_excit[seg])
                else:              # VL or VR -> ventral
                    target = float(ventral_excit[seg])
                # Low-pass filter for smooth activation
                self._muscle_activations[key] = (
                    MUSCLE_FILTER_ALPHA * target
                    + (1 - MUSCLE_FILTER_ALPHA) * self._muscle_activations[key]
                )

        return dict(self._muscle_activations)


# -----------------------------------------------------------------------
# Biologically grounded parameter presets for C. elegans
# -----------------------------------------------------------------------


def _base_params() -> NeuronParameters:
    return NeuronParameters(
        r_base=0.9,
        b_base=1.1,
        c=8,
        lambda_param=15.0,
        p=1.0,
        eta_post=0.005,
        eta_retro=0.005,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.15, 0.04]),
        w_b=np.array([-0.15, 0.04]),
        w_tref=np.array([-15.0, 8.0]),
    )


def _sensory_params() -> NeuronParameters:
    """Sensory neurons: lower threshold, faster integration."""
    return NeuronParameters(
        r_base=0.5,
        b_base=0.7,
        c=4,
        lambda_param=6.0,
        p=1.0,
        eta_post=0.01,
        eta_retro=0.005,
        num_neuromodulators=2,
        gamma=np.array([0.98, 0.99]),
        w_r=np.array([-0.2, 0.05]),
        w_b=np.array([-0.2, 0.05]),
        w_tref=np.array([-12.0, 6.0]),
    )


def _motor_params() -> NeuronParameters:
    """
    Motor neurons: lower threshold so they respond to moderate interneuron
    drive from the locomotion command circuit.
    """
    return NeuronParameters(
        r_base=0.35,
        b_base=0.5,
        c=6,
        lambda_param=8.0,
        p=1.2,
        eta_post=0.01,
        eta_retro=0.01,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.15, 0.04]),
        w_b=np.array([-0.15, 0.04]),
        w_tref=np.array([-10.0, 5.0]),
    )


def _interneuron_params() -> NeuronParameters:
    """Interneurons: high threshold, long integration, serve as CPG."""
    return NeuronParameters(
        r_base=1.0,
        b_base=1.3,
        c=12,
        lambda_param=25.0,
        p=1.0,
        eta_post=0.003,
        eta_retro=0.003,
        num_neuromodulators=2,
        gamma=np.array([0.995, 0.999]),
        w_r=np.array([-0.1, 0.03]),
        w_b=np.array([-0.1, 0.03]),
        w_tref=np.array([-20.0, 10.0]),
    )
