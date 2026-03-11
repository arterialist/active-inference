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
    MOTOR_NEURON_POSITIONS,
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

    def __init__(
        self,
        connectome: ConnectomeData,
        log_level: str = "WARNING",
        enable_m0: bool = True,
        enable_m1: bool = True,
    ):
        self._connectome = connectome
        self._log_level = log_level
        self._enable_m0 = enable_m0
        self._enable_m1 = enable_m1
        self._network: NeuronNetwork | None = None
        self._name_to_id: dict[str, int] = {}

        # Per-muscle activation state (low-pass filtered)
        self._muscle_activations: dict[str, float] = {}
        self._init_muscles()

        # Previous sensory inputs for computing temporal surprise (M0)
        self._prev_sensory: dict[str, float] = {}

        # EMA-smoothed chemosensory signals for dC/dt computation.
        # Filters out undulation-frequency noise so M0/M1 reflects the
        # true gradient, not head oscillation artifacts.
        self._chem_ema: dict[str, float] = {}
        self._prev_chem_ema: dict[str, float] = {}

        # Global neuromodulatory state for volume transmission
        self._global_m0: float = 0.0
        self._global_m1: float = 0.0

        # Build the PAULA network
        self._build()

    # ------------------------------------------------------------------
    # BaseNervousSystem interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        CElegansNervousSystem._seg_map = None
        self._build()
        self._init_muscles()
        self._prev_sensory.clear()
        self._chem_ema.clear()
        self._prev_chem_ema.clear()
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
        if self._enable_m0 or self._enable_m1:
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

    @property
    def neuromod_levels(self) -> tuple[float, float]:
        """Return current global neuromodulator levels (M0, M1)."""
        return (self._global_m0, self._global_m1)

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
            param_overrides=_neuron_param_overrides(),
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
    # K_STRESS reduced so worm gently steers back on course instead of
    # shattering motor rhythm and looping backward.
    _K_STRESS_SYN: float = 5000.0
    _K_REWARD_SYN: float = 8000.0

    _K_VOL_STRESS: float = 1500.0
    _K_VOL_REWARD: float = 2000.0

    # Deadzone: only spike M0 when chemical drops by a significant amount.
    # Tiny dips during head swings are ignored to avoid over-triggering pirouettes.
    _STRESS_DEADZONE: float = 0.015

    _CHEMOSENSORY_NAMES: set[str] = {"ASEL", "ASER", "AWCL", "AWCR"}

    # EMA smoothing for chemosensory dC/dt. Alpha=0.02 → ~50-tick time
    # constant (100ms at 2ms timestep), matching ASE temporal integration.
    _CHEM_EMA_ALPHA: float = 0.02

    # Tonic forward drive: injected into AVB and B-type motor neurons each
    # tick to model the AVB↔B-type gap junction coupling that biases C. elegans
    # toward forward locomotion (~80% of time). The Cook et al. connectome
    # over-represents chemical synapses onto AVA, under-representing the
    # electrical coupling that maintains the forward state.
    _TONIC_FWD_CMD: float = 0.25
    _TONIC_FWD_MOTOR: float = 0.11
    _FWD_CMD_NAMES: set[str] = {"AVBL", "AVBR"}
    _FWD_MOTOR_PREFIXES: tuple[str, ...] = ("DB", "VB")

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

            # For chemosensory neurons, compute delta_c from EMA-smoothed
            # signals to filter out undulation-frequency head oscillations.
            if neuron_name in self._CHEMOSENSORY_NAMES:
                ema = self._chem_ema.get(neuron_name, clamped)
                ema = self._CHEM_EMA_ALPHA * clamped + (1 - self._CHEM_EMA_ALPHA) * ema
                self._chem_ema[neuron_name] = ema
                prev_ema = self._prev_chem_ema.get(neuron_name, ema)
                delta_c = ema - prev_ema
                self._prev_chem_ema[neuron_name] = ema
                delta_accum += delta_c
                n_chem += 1
            else:
                prev = self._prev_sensory.get(neuron_name, clamped)
                delta_c = clamped - prev

            if delta_c < -self._STRESS_DEADZONE:
                excess = abs(delta_c) - self._STRESS_DEADZONE
                m0 = float(np.clip(excess * self._K_STRESS_SYN, 0.0, 5.0)) if self._enable_m0 else 0.0
                m1 = 0.0
            elif delta_c > 0:
                m0 = 0.0
                m1 = float(np.clip(delta_c * self._K_REWARD_SYN, 0.0, 5.0)) if self._enable_m1 else 0.0
            else:
                m0 = 0.0
                m1 = 0.0
            mod = np.array([m0, m1])

            self._network.set_external_input(
                neuron_id=nid,
                synapse_id=synapse_id,
                info=clamped,
                mod=mod,
            )

        self._prev_sensory = dict(sensory_inputs)

        self._inject_tonic_forward()

        avg_delta = delta_accum / n_chem if n_chem > 0 else 0.0
        if avg_delta < -self._STRESS_DEADZONE:
            excess = abs(avg_delta) - self._STRESS_DEADZONE
            self._global_m0 = float(np.clip(
                excess * self._K_VOL_STRESS, 0.0, 2.0
            )) if self._enable_m0 else 0.0
            self._global_m1 = 0.0
        elif avg_delta > 0:
            self._global_m0 = 0.0
            self._global_m1 = float(np.clip(
                avg_delta * self._K_VOL_REWARD, 0.0, 2.0
            )) if self._enable_m1 else 0.0
        else:
            self._global_m0 = 0.0
            self._global_m1 = 0.0

    def _inject_tonic_forward(self) -> None:
        """Inject tonic depolarizing current into forward-circuit neurons.

        Models the AVB↔B-motor gap junction coupling that maintains
        forward locomotion as the default behavioral state.  The injected
        current is additive to whatever synaptic input the neuron receives.
        """
        if self._network is None:
            return
        for name in self._FWD_CMD_NAMES:
            nid = self._name_to_id.get(name)
            if nid is None:
                continue
            n = self._network.network.neurons.get(nid)
            if n is not None:
                n.S += self._TONIC_FWD_CMD

        for name, nid in self._name_to_id.items():
            prefix = name.rstrip("0123456789")
            if prefix in self._FWD_MOTOR_PREFIXES:
                n = self._network.network.neurons.get(nid)
                if n is not None:
                    n.S += self._TONIC_FWD_MOTOR

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
            if self._enable_m0 and self._global_m0 > 1e-8:
                neuron.M_vector[0] += (1.0 - gamma[0]) * self._global_m0
            if self._enable_m1 and self._global_m1 > 1e-8:
                neuron.M_vector[1] += (1.0 - gamma[1]) * self._global_m1

    # Graded output normalization.
    # C. elegans motor neurons transmit via graded potentials, not spikes.
    _S_NORM: float = 0.25
    _INHIB_WEIGHT: float = 0.3
    _RECIP_INHIB: float = 0.5
    _FWD_WEIGHT: float = 1.0
    _BKW_WEIGHT: float = 0.3

    # Precomputed once: which motor neurons contribute to which segments.
    # Built lazily on first call via _build_segment_map().
    _seg_map: dict[str, list[tuple[str, float]]] | None = None

    @classmethod
    def _build_segment_map(cls) -> dict[str, list[tuple[str, float]]]:
        """Map each motor neuron to its anatomically nearest segment(s).

        Gaussian spillover (sigma=0.8 segments) avoids hard boundaries
        and lets neurons with positions between two segments drive both
        proportionally.

        Note: the MuJoCo body chain extends in -x (seg0=head at +x,
        seg12=tail at -x).  The worm's natural locomotion direction is
        -x.  Biological forward = model -x direction.
        """
        sigma = 0.8
        seg_map: dict[str, list[tuple[str, float]]] = {
            f"seg{s}": [] for s in range(N_BODY_SEGMENTS)
        }
        for name, frac in MOTOR_NEURON_POSITIONS.items():
            center_seg = frac * (N_BODY_SEGMENTS - 1)
            for seg in range(N_BODY_SEGMENTS):
                dist = abs(seg - center_seg)
                if dist > 3.0:
                    continue
                w = float(np.exp(-0.5 * (dist / sigma) ** 2))
                if w > 0.01:
                    seg_map[f"seg{seg}"].append((name, w))
        return seg_map

    def _decode_motor_outputs(self, current_tick: int) -> dict[str, float]:
        """
        Convert motor neuron membrane potentials to muscle activations.

        Uses graded (non-spiking) readout of motor neuron S values with
        anatomically accurate segment mapping from MOTOR_NEURON_POSITIONS.

        Motor neuron roles (White et al. 1986):
          DB/DA + AS → dorsal excitation (B=forward, A=backward)
          VB/VA      → ventral excitation (B=forward, A=backward)
          DD         → dorsal GABAergic inhibition
          VD         → ventral GABAergic inhibition
        """
        if self._seg_map is None:
            CElegansNervousSystem._seg_map = self._build_segment_map()

        def _graded(name: str) -> float:
            n = self.get_neuron_by_name(name)
            if n is None:
                return 0.0
            return float(np.clip(n.S / self._S_NORM, 0.0, 1.0))

        dorsal_excit = np.zeros(N_BODY_SEGMENTS)
        ventral_excit = np.zeros(N_BODY_SEGMENTS)

        for seg in range(N_BODY_SEGMENTS):
            contributors = self._seg_map[f"seg{seg}"]
            d_exc = 0.0
            v_exc = 0.0
            d_inh = 0.0
            v_inh = 0.0
            for name, w in contributors:
                g = _graded(name) * w
                prefix = name.rstrip("0123456789")
                if prefix == "DB":
                    d_exc += g * self._FWD_WEIGHT
                elif prefix == "DA":
                    d_exc += g * self._BKW_WEIGHT
                elif prefix == "AS":
                    d_exc += g * 0.5
                elif prefix == "VB":
                    v_exc += g * self._FWD_WEIGHT
                elif prefix == "VA":
                    v_exc += g * self._BKW_WEIGHT
                elif prefix == "DD":
                    d_inh += g
                elif prefix == "VD":
                    v_inh += g

            dorsal_excit[seg] = d_exc - d_inh * self._INHIB_WEIGHT
            ventral_excit[seg] = v_exc - v_inh * self._INHIB_WEIGHT

        dorsal_excit = np.clip(dorsal_excit, 0.0, 1.0)
        ventral_excit = np.clip(ventral_excit, 0.0, 1.0)

        # Reciprocal inhibition (mechanical antagonism of body-wall muscles)
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
    """Default motor neurons (excitatory B/A-type cholinergic).

    Low threshold so they fire readily from command interneuron drive.
    High neuromodulatory sensitivity (tyramine/octopamine in vivo).
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


def _motor_inhib_params() -> NeuronParameters:
    """GABAergic inhibitory motor neurons (DD, VD).

    Lower threshold and shorter time constant than excitatory motor
    neurons: cross-inhibition must respond quickly to track the
    excitatory drive and produce D-V alternation.
    """
    return NeuronParameters(
        r_base=0.15,
        b_base=0.25,
        c=3,
        lambda_param=4.0,
        p=1.0,
        eta_post=0.002,
        eta_retro=0.002,
        num_neuromodulators=2,
        gamma=np.array([0.99, 0.995]),
        w_r=np.array([-0.2, 0.05]),
        w_b=np.array([-0.2, 0.05]),
        w_tref=np.array([20.0, -10.0]),
    )


def _interneuron_params() -> NeuronParameters:
    """Local interneurons: moderate threshold, relay and integrate."""
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


def _command_fwd_params() -> NeuronParameters:
    """Forward command interneurons (AVB, PVC).

    Lower threshold than backward commands: C. elegans spends ~80% of
    its time in forward locomotion (Zhao et al. 2003).  AVB/PVC
    should be tonically active at baseline.
    """
    return NeuronParameters(
        r_base=0.4,
        b_base=0.6,
        c=8,
        lambda_param=10.0,
        p=1.0,
        eta_post=0.001,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.995, 0.998]),
        w_r=np.array([-0.1, 0.03]),
        w_b=np.array([-0.1, 0.03]),
        w_tref=np.array([12.0, -6.0]),
    )


def _command_bkw_params() -> NeuronParameters:
    """Backward command interneurons (AVA, AVD).

    Higher threshold: only activated by nociceptive or strong aversive
    stimuli, then briefly dominates forward circuit.
    """
    return NeuronParameters(
        r_base=0.85,
        b_base=1.1,
        c=12,
        lambda_param=16.0,
        p=1.0,
        eta_post=0.001,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.995, 0.998]),
        w_r=np.array([-0.1, 0.03]),
        w_b=np.array([-0.1, 0.03]),
        w_tref=np.array([12.0, -6.0]),
    )


def _neuron_param_overrides() -> dict[str, NeuronParameters]:
    """Per-neuron-name parameter overrides for biologically distinct sub-classes."""
    overrides: dict[str, NeuronParameters] = {}

    inhib = _motor_inhib_params()
    for name in [f"DD{i}" for i in range(1, 7)] + [f"VD{i}" for i in range(1, 14)]:
        overrides[name] = inhib

    fwd = _command_fwd_params()
    for name in ["AVBL", "AVBR", "PVCL", "PVCR"]:
        overrides[name] = fwd

    bkw = _command_bkw_params()
    for name in ["AVAL", "AVAR", "AVDL", "AVDR"]:
        overrides[name] = bkw

    return overrides
