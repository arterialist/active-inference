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

from typing import Any

import numpy as np
from loguru import logger

from simulations.base_nervous_system import BaseNervousSystem
from simulations.connectome_loader import ConnectomeData, build_paula_network
from simulations.paula_loader import ensure_paula_available
from simulations.c_elegans.config import (
    MUSCLE_FILTER_ALPHA,
    N_BODY_SEGMENTS,
    MUSCLE_QUADRANTS,
)

ensure_paula_available()
from neuron.neuron import NeuronParameters  # noqa: E402
from neuron.network import NeuronNetwork  # noqa: E402


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

        # Previous sensory inputs for computing temporal surprise (M0)
        self._prev_sensory: dict[str, float] = {}

        # Global neuromodulatory state for volume transmission
        self._global_m0: float = 0.0
        self._global_m1: float = 0.0

        # Build the PAULA network
        self._build()

    # ------------------------------------------------------------------
    # BaseNervousSystem interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._build()
        self._init_muscles()
        self._prev_sensory.clear()
        self._global_m0 = 0.0
        self._global_m1 = 0.0

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

        # --- inject sensory inputs + compute global M0/M1 ---
        self._inject_sensory(sensory_inputs, current_tick)

        # --- run one tick ---
        self._network.run_tick()

        # --- volume transmission: broadcast M0/M1 to all neurons ---
        self._volume_broadcast()

        # --- decode motor outputs ---
        return self._decode_motor_outputs(current_tick)

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
            weight_max=5.0,
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

    # Gain for synapse-level mod injection into sensory neurons.
    _K_STRESS_SYN: float = 5000.0
    _K_REWARD_SYN: float = 3000.0

    # Gain for volume transmission (directly into M_vector of all neurons).
    # dC/dt ≈ 0.0006/tick → volume_target = dC * K_VOL → each tick we add
    # volume_target * (1-gamma) to M_vector so it reaches volume_target at
    # steady state.
    _K_VOL_STRESS: float = 1500.0
    _K_VOL_REWARD: float = 800.0

    _CHEMOSENSORY_NAMES: set[str] = {"ASEL", "ASER", "AWCL", "AWCR"}

    def _inject_sensory(
        self, sensory_inputs: dict[str, float], current_tick: int
    ) -> None:
        """Inject sensory info + compute global neuromod from dC/dt.

        Strictly differential per ALERM Eq 4-5:
        - dC/dt < 0 → M0 (stress, broadens t_ref, pirouette)
        - dC/dt > 0 → M1 (reward, narrows t_ref, crystallise run)

        The global M0/M1 are stored in self._global_m0/m1 and
        broadcast to all neurons via _volume_broadcast after the
        network tick (volume transmission).
        """
        delta_accum = 0.0
        n_chem = 0

        for neuron_name, intensity in sensory_inputs.items():
            nid = self._name_to_id.get(neuron_name)
            if nid is None:
                continue
            neuron = self._network.network.neurons.get(nid)
            if neuron is None or neuron.params.num_inputs == 0:
                continue

            synapse_id = 0
            if synapse_id >= neuron.params.num_inputs:
                continue

            clamped = float(np.clip(intensity, 0.0, 2.0))
            prev = self._prev_sensory.get(neuron_name, clamped)
            delta_c = clamped - prev

            # Accumulate dC/dt from chemosensory neurons for global signal
            if neuron_name in self._CHEMOSENSORY_NAMES:
                delta_accum += delta_c
                n_chem += 1

            # Synapse-level mod for this sensory neuron
            if delta_c < 0:
                m0 = float(np.clip(abs(delta_c) * self._K_STRESS_SYN, 0.0, 5.0))
                m1 = 0.0
            elif delta_c > 0:
                m0 = 0.0
                m1 = float(np.clip(delta_c * self._K_REWARD_SYN, 0.0, 5.0))
            else:
                m0 = 0.0
                m1 = 0.0

            self._network.set_external_input(
                neuron_id=nid,
                synapse_id=synapse_id,
                info=clamped,
                mod=np.array([m0, m1]),
            )

        self._prev_sensory = dict(sensory_inputs)

        # Compute global neuromodulatory drive from aggregate dC/dt
        avg_delta = delta_accum / n_chem if n_chem > 0 else 0.0
        if avg_delta < 0:
            self._global_m0 = float(np.clip(
                abs(avg_delta) * self._K_VOL_STRESS, 0.0, 2.0
            ))
            self._global_m1 = 0.0
        elif avg_delta > 0:
            self._global_m0 = 0.0
            self._global_m1 = float(np.clip(
                avg_delta * self._K_VOL_REWARD, 0.0, 2.0
            ))
        else:
            self._global_m0 = 0.0
            self._global_m1 = 0.0

    def _volume_broadcast(self) -> None:
        """ALERM volume transmission: inject global M0/M1 into all neurons.

        Adds (1-gamma_k) * target_k to each neuron's M_vector[k] so that
        repeated application converges to target_k at steady state through
        the existing gamma-leaky integration.
        """
        if self._network is None:
            return
        if self._global_m0 < 1e-8 and self._global_m1 < 1e-8:
            return
        for neuron in self._network.network.neurons.values():
            gamma = neuron.params.gamma
            if self._global_m0 > 1e-8:
                neuron.M_vector[0] += (1.0 - gamma[0]) * self._global_m0
            if self._global_m1 > 1e-8:
                neuron.M_vector[1] += (1.0 - gamma[1]) * self._global_m1

    # Graded output normalization for motor neurons.
    # C. elegans motor neurons transmit via graded potentials, not spikes.
    # S / _S_NORM maps typical motor neuron S values (~0.05-0.15) into a
    # usable [0, 1] range.  Inhibitory D-type weight is reduced because
    # GABAergic NMJs produce smaller postsynaptic currents than cholinergic.
    _S_NORM: float = 0.25
    _INHIB_WEIGHT: float = 0.3
    _RECIP_INHIB: float = 0.5

    def _decode_motor_outputs(self, current_tick: int) -> dict[str, float]:
        """
        Convert motor neuron membrane potentials to muscle activations.

        Uses graded (non-spiking) readout of motor neuron S values.
        B-type (DB, VB) excite their respective muscle side; D-type
        inhibit the side they innervate (DD -> dorsal, VD -> ventral).
        No artificial phase-lag buffer: the traveling wave must emerge
        from PAULA network dynamics and connectome topology.
        """
        db_neurons = [f"DB{i}" for i in range(1, 8)]    # 7 dorsal B-type
        vb_neurons = [f"VB{i}" for i in range(1, 12)]   # 11 ventral B-type
        dd_neurons = [f"DD{i}" for i in range(1, 7)]    # 6 dorsal D-type
        vd_neurons = [f"VD{i}" for i in range(1, 14)]   # 13 ventral D-type
        as_neurons = [f"AS{i}" for i in range(1, 12)]   # 11 A-type

        def _graded(name: str) -> float:
            """Graded motor output proportional to membrane depolarization."""
            n = self.get_neuron_by_name(name)
            if n is None:
                return 0.0
            return float(np.clip(n.S / self._S_NORM, 0.0, 1.0))

        dorsal_excit = np.zeros(N_BODY_SEGMENTS)
        ventral_excit = np.zeros(N_BODY_SEGMENTS)

        for seg in range(N_BODY_SEGMENTS):
            db_idx = min(int(seg * len(db_neurons) / N_BODY_SEGMENTS), len(db_neurons) - 1)
            dorsal_excit[seg] += _graded(db_neurons[db_idx])

            as_idx = min(int(seg * len(as_neurons) / N_BODY_SEGMENTS), len(as_neurons) - 1)
            dorsal_excit[seg] += _graded(as_neurons[as_idx]) * 0.5

            vb_idx = min(int(seg * len(vb_neurons) / N_BODY_SEGMENTS), len(vb_neurons) - 1)
            ventral_excit[seg] += _graded(vb_neurons[vb_idx])

            # DD innervates DORSAL body-wall muscles (GABAergic inhibition)
            dd_idx = min(int(seg * len(dd_neurons) / N_BODY_SEGMENTS), len(dd_neurons) - 1)
            dorsal_excit[seg] -= _graded(dd_neurons[dd_idx]) * self._INHIB_WEIGHT

            # VD innervates VENTRAL body-wall muscles (GABAergic inhibition)
            vd_idx = min(int(seg * len(vd_neurons) / N_BODY_SEGMENTS), len(vd_neurons) - 1)
            ventral_excit[seg] -= _graded(vd_neurons[vd_idx]) * self._INHIB_WEIGHT

        dorsal_excit = np.clip(dorsal_excit, 0.0, 1.0)
        ventral_excit = np.clip(ventral_excit, 0.0, 1.0)

        # Reciprocal inhibition: dorsal and ventral body-wall muscles are
        # mechanical antagonists.  Whichever side has stronger neural drive
        # suppresses the other, creating the D-V alternation needed for
        # bending.  This supplements the DD/VD cross-inhibition pathway.
        d_push = dorsal_excit - ventral_excit * self._RECIP_INHIB
        v_push = ventral_excit - dorsal_excit * self._RECIP_INHIB
        dorsal_excit = np.clip(d_push, 0.0, 1.0)
        ventral_excit = np.clip(v_push, 0.0, 1.0)

        for seg in range(N_BODY_SEGMENTS):
            for quad in MUSCLE_QUADRANTS:
                key = f"seg{seg}_{quad}"
                target = float(dorsal_excit[seg]) if "D" in quad else float(ventral_excit[seg])
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
        eta_post=0.001,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.15, 0.04]),
        w_b=np.array([-0.15, 0.04]),
        # ALERM Eq 4-5: M0 broadens t_ref (+), M1 narrows it (-)
        w_tref=np.array([15.0, -8.0]),
    )


def _sensory_params() -> NeuronParameters:
    """Sensory neurons: lowest threshold, fastest integration."""
    return NeuronParameters(
        r_base=0.5,
        b_base=0.7,
        c=4,
        lambda_param=6.0,
        p=1.0,
        eta_post=0.002,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.98, 0.99]),
        w_r=np.array([-0.2, 0.05]),
        w_b=np.array([-0.2, 0.05]),
        w_tref=np.array([12.0, -6.0]),
    )


def _motor_params() -> NeuronParameters:
    """Motor neurons: low threshold, fast integration.

    In the real worm, body motor neurons fire readily from sparse
    command interneuron input + gap-junction coupling along the
    ventral nerve cord.  Motor neurons are highly sensitive to
    neuromodulatory volume transmission (tyramine/octopamine).
    """
    return NeuronParameters(
        r_base=0.2,
        b_base=0.3,
        c=4,
        lambda_param=6.0,
        p=1.2,
        eta_post=0.002,
        eta_retro=0.002,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.25, 0.06]),
        w_b=np.array([-0.25, 0.06]),
        w_tref=np.array([30.0, -15.0]),
    )


def _interneuron_params() -> NeuronParameters:
    """Interneurons: moderate threshold, faster integration to relay sensory drive."""
    return NeuronParameters(
        r_base=0.6,
        b_base=0.85,
        c=8,
        lambda_param=10.0,
        p=1.0,
        eta_post=0.001,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.15, 0.04]),
        w_b=np.array([-0.15, 0.04]),
        w_tref=np.array([15.0, -8.0]),
    )
