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

import heapq
from dataclasses import replace
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
        evol_config: dict[str, Any] | None = None,
    ):
        self._connectome = connectome
        self._log_level = log_level
        self._enable_m0 = enable_m0
        self._enable_m1 = enable_m1
        self._evol_config = evol_config or {}

        # Apply neuromod config overrides (instance attrs override class defaults)
        for key in (
            "K_STRESS_SYN",
            "K_REWARD_SYN",
            "K_VOL_STRESS",
            "K_VOL_REWARD",
            "STRESS_DEADZONE",
            "CHEM_EMA_ALPHA_FAST",
            "CHEM_EMA_ALPHA_SLOW",
            "TONIC_FWD_CMD",
            "TONIC_FWD_MOTOR",
            "K_OFF_SUPPRESS",
            "TONIC_OFF_CELL",
            "PROPRIO_MOTOR_GAIN",
            "PROPRIO_TAIL_DECAY",
        ):
            if key in self._evol_config:
                setattr(self, f"_{key}", self._evol_config[key])
        # Backward compat: legacy single CHEM_EMA_ALPHA maps to slow filter
        if (
            "CHEM_EMA_ALPHA" in self._evol_config
            and "CHEM_EMA_ALPHA_SLOW" not in self._evol_config
        ):
            self._CHEM_EMA_ALPHA_SLOW = self._evol_config["CHEM_EMA_ALPHA"]
        self._network: NeuronNetwork | None = None
        self._name_to_id: dict[str, int] = {}

        # Per-muscle activation state (low-pass filtered)
        self._muscle_activations: dict[str, float] = {}
        self._init_muscles()

        # Previous sensory inputs for computing temporal surprise (M0)
        self._prev_sensory: dict[str, float] = {}

        # Dual-EMA bandpass filter state for chemosensory dC/dt.
        # Fast EMA tracks head-sweep; slow EMA tracks environmental gradient.
        # delta_c = fast - slow cancels undulation noise.
        self._chem_ema_fast: dict[str, float] = {}
        self._chem_ema_slow: dict[str, float] = {}

        # Global neuromodulatory state for volume transmission
        self._global_m0: float = 0.0
        self._global_m1: float = 0.0

        # Build the PAULA network
        self._build()

    # ------------------------------------------------------------------
    # BaseNervousSystem interface
    # ------------------------------------------------------------------

    def reset(self, *, rebuild_network: bool = True) -> None:
        from simulations import evol_trace

        CElegansNervousSystem._seg_map = None
        with evol_trace.span("reset_nervous_rebuild" if rebuild_network else "reset_nervous_light"):
            if rebuild_network:
                self._build()
            elif self._network is not None:
                self._network.reset_simulation()
            else:
                self._build()
        self._init_muscles()
        self._prev_sensory.clear()
        self._chem_ema_fast.clear()
        self._chem_ema_slow.clear()
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

        # --- bound graded motor S (no spike reset → integrator must be capped)
        self._clamp_motor_S()

        # --- graded synaptic release for body-wall motor neurons ---
        # Real C. elegans body-wall neurons (B/A/D-types) signal via continuous
        # graded transmission, not spikes (Liu 2009; Beg & Jorgensen 2003).
        # PAULA only releases on spikes, so when we suppress firing we must
        # supply chemical drive directly.
        self._inject_graded_release()

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

    def get_neuron_names_paula_order(self) -> list[str]:
        """Biological names ordered by PAULA id (0 .. n-1), matching network dict keys."""
        if self._network is None:
            return []
        neurons = self._network.network.neurons
        return [
            str(neurons[i].metadata.get("name", str(i)))
            for i in sorted(neurons.keys(), key=int)
        ]

    def get_compact_neural_snapshot(self) -> tuple[list[float], list[int], list[float]]:
        """Lightweight S / fired / primary threshold ``r`` in paula_id order (no per-name dict).

        Avoids building the large string-key dict used by :meth:`get_neuron_states`
        when only dense vectors are needed for streaming or logging.
        """
        if self._network is None:
            return [], [], []
        neurons = self._network.network.neurons
        if not neurons:
            return [], [], []
        ids = sorted(neurons.keys(), key=int)
        n = int(ids[-1]) + 1
        s_out = [0.0] * n
        f_out = [0] * n
        r_out = [0.0] * n
        for i in ids:
            neuron = neurons[i]
            ii = int(i)
            if 0 <= ii < n:
                s_out[ii] = float(neuron.S)
                f_out[ii] = 1 if float(neuron.O) > 0 else 0
                r_out[ii] = float(neuron.r)
        return s_out, f_out, r_out

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

    def _apply_evol_params(
        self,
        params: NeuronParameters,
        name: str,
    ) -> NeuronParameters:
        """Apply evol_config overrides to neuron params."""
        overrides = self._evol_config.get("neuron_params", {}).get(name)
        if not overrides:
            return params
        kwargs = {}
        for k, v in overrides.items():
            if hasattr(params, k):
                kwargs[k] = (
                    np.array(v)
                    if k in ("gamma", "w_r", "w_b", "w_tref")
                    and isinstance(v, (list, tuple))
                    else v
                )
        return replace(params, **kwargs) if kwargs else params

    # Neurons that signal via graded transmission rather than spikes.
    # In real C. elegans almost the entire nervous system is graded; only a
    # handful of cells (notably AVA in some conditions, ASE bursting on dC/dt)
    # show clear spike-like events. We treat as graded:
    #   - Body wall motors (B/A/D types): Liu 2009; Goodman 2012
    #   - Head ring motors (RMD/SMD/SMB): Mulcahy 2018, Lockery & Goodman 2009
    # Body-wall list is what gets spikes suppressed AND graded release applied.
    _BODY_WALL_PREFIXES: tuple[str, ...] = ("DB", "VB", "DA", "VA", "AS", "DD", "VD")
    # Names (not just prefixes) of head-ring graded sources. Listed explicitly
    # because RMD/SMD/SMB don't follow the digit-suffix prefix pattern.
    _HEAD_RING_GRADED: tuple[str, ...] = (
        "RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMDL", "RMDR",
        "SMDDL", "SMDDR", "SMDVL", "SMDVR",
        "SMBDL", "SMBDR", "SMBVL", "SMBVR",
    )
    _graded_motor_only: bool = True  # toggle exposed via lab UI

    def _is_graded_neuron(self, name: str) -> bool:
        """Whether a neuron uses graded (continuous) transmission instead of
        PAULA's spike-based release. Includes body-wall motors and head-ring
        motors — covers most of the locomotion connectome."""
        prefix = name.rstrip("0123456789")
        if prefix in self._BODY_WALL_PREFIXES:
            return True
        if name in self._HEAD_RING_GRADED:
            return True
        return False

    def _suppress_motor_spikes(self, threshold: float = 1e6) -> None:
        """Set r_base/b_base on graded (non-spiking) neurons to a value far above
        any reachable membrane potential, forcing graded-only transmission."""
        if self._network is None:
            return
        for name, nid in self._name_to_id.items():
            if self._is_graded_neuron(name):
                neuron = self._network.network.neurons.get(nid)
                if neuron is None:
                    continue
                neuron.params.r_base = float(threshold)
                neuron.params.b_base = float(threshold)
                neuron.r = float(threshold)
                neuron.b = float(threshold)

    # Graded-synapse release table: list of (target_id, synapse_slot, source_id, weight)
    # where source is a body-wall motor neuron. Built lazily after network is up.
    _graded_links: list[tuple[int, int, int, float]] | None = None

    def _build_graded_link_table(self) -> list[tuple[int, int, int, float]]:
        """Catalogue chemical synapses where the presynaptic side is a graded
        (non-spiking) neuron. Each tick we inject pre.S × weight directly into
        the target so the rhythm propagates continuously through the connectome
        without PAULA's spike-quantal attenuation."""
        if self._network is None:
            return []
        graded_ids = set()
        for name, nid in self._name_to_id.items():
            if self._is_graded_neuron(name):
                graded_ids.add(nid)

        links: list[tuple[int, int, int, float]] = []
        for target_id, target_neuron in self._network.network.neurons.items():
            for slot_id, source_tuple in target_neuron.synapse_sources.items():
                if not source_tuple:
                    continue
                source_id, _terminal_id = source_tuple
                if source_id not in graded_ids:
                    continue
                pt = target_neuron.postsynaptic_points.get(slot_id)
                if pt is None:
                    continue
                base_weight = float(pt.u_i.info)
                links.append((int(target_id), int(slot_id), int(source_id), base_weight))
        return links

    # Per-tick gain on graded synaptic drive: scales pre.S∈[0,1] × weight
    # into a small additive S contribution. Tuned so a fully-active source
    # at full weight contributes ~0.01-0.05 to target.S each tick.
    _GRADED_RELEASE_GAIN: float = 0.005

    # Soft bound on body-wall motor S to prevent runaway in absence of firing
    # reset. Real C. elegans body-wall neurons stay within ~10-20 mV of rest
    # (Goodman et al. 2012); this clip simulates intrinsic conductance balance.
    _GRADED_S_BOUND: float = 1.0

    def _clamp_motor_S(self) -> None:
        """Bound graded-neuron S to ±_GRADED_S_BOUND. Without spikes there's
        no native reset; the leaky integrator alone lets S drift to ±1000.
        Real C. elegans body-wall membrane potentials operate in a narrow
        ~mV range due to channel-dependent shunting (Goodman 2012)."""
        if self._network is None:
            return
        bound = float(self._GRADED_S_BOUND)
        for name, nid in self._name_to_id.items():
            if self._is_graded_neuron(name):
                neuron = self._network.network.neurons.get(nid)
                if neuron is not None:
                    if neuron.S > bound:
                        neuron.S = bound
                    elif neuron.S < -bound:
                        neuron.S = -bound

    def _inject_graded_release(self) -> None:
        """Inject graded chemical release for body-wall motor synapses.

        Real C. elegans body-wall motor neurons release continuously in
        proportion to membrane depolarization (Liu 2009; Beg & Jorgensen 2003).
        PAULA's release is spike-quantal, so when we suppress spikes (graded mode)
        we instead add a small continuous S contribution to each postsynaptic
        target proportional to source.S × synapse weight.

        Contribution: gain × clip(pre.S / S_NORM, 0, 1) × weight added to
        target.S each tick. Negative pre.S is treated as zero release
        (hyperpolarization doesn't release transmitter)."""
        if self._network is None or not self._graded_motor_only:
            return
        if self._graded_links is None:
            self._graded_links = self._build_graded_link_table()
        gain = float(self._GRADED_RELEASE_GAIN)
        norm = self._S_NORM
        neurons = self._network.network.neurons
        for target_id, _slot, source_id, weight in self._graded_links:
            src = neurons.get(source_id)
            tgt = neurons.get(target_id)
            if src is None or tgt is None:
                continue
            release = max(0.0, src.S) / norm if norm > 0 else max(0.0, src.S)
            # Cap release at 1.0 to mimic saturated neurotransmitter pool.
            if release > 1.0: release = 1.0
            # weight is signed (excitatory positive, inhibitory negative).
            tgt.S += gain * release * weight

    def _build(self) -> None:
        """(Re)build the PAULA NeuronNetwork from the connectome."""
        base = self._apply_evol_params(_base_params(), "base")
        sensory = self._apply_evol_params(_sensory_params(), "sensory")
        motor = self._apply_evol_params(_motor_params(), "motor")
        interneuron = self._apply_evol_params(_interneuron_params(), "interneuron")

        def _evol_key(name: str) -> str:
            if name.startswith(("DD", "VD")):
                return "motor_inhib"
            if name in ("AVBL", "AVBR", "PVCL", "PVCR"):
                return "command_fwd"
            if name in self._OFF_CELL_NAMES:
                return "off_cell"
            return "command_bkw"

        overrides_raw = _neuron_param_overrides()
        overrides = {
            k: self._apply_evol_params(v, _evol_key(k))
            for k, v in overrides_raw.items()
        }
        network, name_to_id = build_paula_network(
            self._connectome,
            base_params=base,
            sensory_params=sensory,
            motor_params=motor,
            interneuron_params=interneuron,
            weight_max=5.0,
            log_level=self._log_level,
            param_overrides=overrides,
        )
        self._network = network
        self._name_to_id = name_to_id
        # Body-wall motor neurons (DB/VB/DA/VA/AS/DD/VD) signal via graded
        # potentials in real C. elegans (Liu et al. 2009; Goodman et al. 2012).
        # PAULA defaults give them spike thresholds; we suppress firing here so
        # graded readout is clean (no S resets per spike) and the connectome
        # behaves like the biological body-wall system.
        if getattr(self, "_graded_motor_only", True):
            self._suppress_motor_spikes()
        # Cached link table is built lazily on first tick; invalidate here so
        # any rebuild produces a fresh map matching the current connectome.
        self._graded_links = None
        # Disable PAULA's synaptic plasticity (eta_post, eta_retro). At the
        # default 0.01 rate, weights drift on second-timescales — accumulates
        # over thousands of ticks and retunes the connectome away from its
        # initial calibration, eventually saturating critical pathways
        # (e.g. RIAR → RMD that pulls RMD into clamp). Real C. elegans
        # plasticity operates on hour/day timescales, not seconds.
        if getattr(self, "_plasticity_disabled", True):
            self._disable_plasticity()
        logger.info(
            f"CElegansNervousSystem ready: {len(name_to_id)} neurons in PAULA network"
        )

    def _disable_plasticity(self) -> None:
        """Set eta_post = eta_retro = 0 on every neuron, freezing synaptic
        weights at their initial connectome-derived values."""
        if self._network is None:
            return
        for nid, neuron in self._network.network.neurons.items():
            neuron.params.eta_post = 0.0
            neuron.params.eta_retro = 0.0

    def _init_muscles(self) -> None:
        """Initialise per-muscle activation state to zero."""
        self._muscle_activations = {}
        for seg in range(N_BODY_SEGMENTS):
            for quad in MUSCLE_QUADRANTS:
                self._muscle_activations[f"seg{seg}_{quad}"] = 0.0

    # Gain for synapse-level mod injection into sensory neurons.
    # K_STRESS reduced so worm gently steers back on course instead of
    # shattering motor rhythm and looping backward.
    _K_STRESS_SYN: float = 4000.0
    _K_REWARD_SYN: float = 4000.0

    _K_VOL_STRESS: float = 2000.0
    _K_VOL_REWARD: float = 2000.0

    # Deadzone: only spike M0 when chemical drops by a significant amount.
    # Tiny dips during head swings are ignored to avoid over-triggering pirouettes.
    _STRESS_DEADZONE: float = 0.00005

    _CHEMOSENSORY_NAMES: set[str] = {"ASEL", "ASER", "AWCL", "AWCR"}

    # Dual-EMA bandpass filter for chemosensory dC/dt.
    # Biological basis: AWC/ASE interneurons act as bandpass filters,
    # using two temporal integration loops to subtract head-swing noise
    # from the environmental gradient signal.
    #
    # Fast EMA tracks immediate head-sweep (~10 ticks, alpha~0.2).
    # Slow EMA tracks long-term environmental gradient (~40+ ticks).
    # delta_c = fast - slow cancels undulation-frequency oscillations,
    # leaving only the true navigational gradient.
    _CHEM_EMA_ALPHA_FAST: float = 0.2
    _CHEM_EMA_ALPHA_SLOW: float = 0.01

    # Tonic forward drive: injected into AVB and B-type motor neurons each
    # tick to model the AVB↔B-type gap junction coupling that biases C. elegans
    # toward forward locomotion (~80% of time). The Cook et al. connectome
    # over-represents chemical synapses onto AVA, under-representing the
    # electrical coupling that maintains the forward state.
    _TONIC_FWD_CMD: float = 0.30
    _TONIC_FWD_MOTOR: float = 0.0
    # Both AVB↔B-motor gap junctions AND the PVC→B chemical chain are required
    # for tonic forward state in C. elegans (Chalfie 1985; Wicks 1996; Faumont 2011).
    # Without PVC tonic drive, DB neurons receive no gap-junction-equivalent and
    # the body falls into a strong ventral curl driven by VB sensory pathways.
    _FWD_CMD_NAMES: set[str] = {"AVBL", "AVBR", "PVCL", "PVCR"}
    _FWD_MOTOR_PREFIXES: tuple[str, ...] = ("DB", "VB")

    # Backward-locomotion command interneurons (AVA/AVD/AVE).
    # Real C. elegans switches between forward and backward states stochastically
    # (Pierce-Shimomura 1999; Roberts et al. 2016; Gordus et al. 2015). The
    # switching is driven by intrinsic neuronal noise (synaptic vesicle release
    # jitter, channel noise) that occasionally depolarises AVA past threshold,
    # triggering a reversal episode.
    _BKW_CMD_NAMES: set[str] = {"AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"}
    # Noise σ on command interneurons (Ornstein-Uhlenbeck per tick) — tunable.
    _CMD_NOISE_SIGMA: float = 0.05
    _CMD_NOISE_TAU: float = 50.0   # ticks (~100 ms correlation time)
    # OU state for command interneuron noise.
    _cmd_noise_state: dict[str, float] = {}

    # Reversal latch: AVA plateau-potential dynamics. Trigger when AVA mean S
    # crosses a HIGH threshold (must be a strong burst, not just noise jitter).
    # Bio reversal duration 1-3s, refractory 2-5s (Pierce-Shimomura 1999).
    # Set trigger to 1e9 to disable.
    _REV_LATCH_TRIGGER_S: float = 1.50
    _REV_LATCH_DURATION_TICKS_RANGE: tuple[int, int] = (500, 1500)
    _REV_REFRACTORY_TICKS_RANGE: tuple[int, int] = (1000, 2500)
    _rev_state: str = "idle"
    _rev_state_until_tick: int = 0
    _rev_current_tick: int = 0

    # OFF-cell neurons (AWC, ASER): tonically active, suppressed during
    # stimulus, burst on removal (Chalasani et al. 2007, Suzuki et al. 2008).
    _OFF_CELL_NAMES: set[str] = {"AWCL", "AWCR", "ASER"}
    _K_OFF_SUPPRESS: float = 5.0  # gain: absolute concentration → M1
    _TONIC_OFF_CELL: float = 0.15  # tonic S baseline for OFF-cell firing

    # B-type motor neuron proprioception gain (Wen et al. 2012).
    _PROPRIO_MOTOR_GAIN: float = 0.10

    # Anterior→posterior taper on the proprioceptive gain. With a
    # uniform gain every segment amplifies its own bend through the
    # stretch-reflex chain, so the wave saturates in mid-body and
    # reflects off the tail as a standing wave. Tapering the gain
    # toward the tail lets the head drive the wave while the
    # posterior passively follows and doesn't lock. 0.0 disables the
    # taper; 1.0 means the last motor neuron receives no
    # proprioceptive drive.
    _PROPRIO_TAIL_DECAY: float = 0.7

    # Head central pattern generator (CPG).
    # Real C. elegans relies on a head rhythm circuit (RIM/RMD/SMD/AVB
    # oscillatory subnet) to seed the locomotion wave. The simplified
    # PAULA connectome has no intrinsic oscillator, so the body alone
    # cannot break symmetry and produces a synchronous whole-body
    # contraction (verified — every adjacent segment's motor drive
    # correlated at r>0.5 with 0-lag in muscle_wave analysis).
    #
    # We inject a sinusoidal current into the anterior-most DB/VB
    # neurons in anti-phase. The rest of the chain picks up this
    # rhythm through the signed proprioceptive stretch reflex
    # (MOTOR_PROPRIO anterior offset), producing a posterior-traveling wave
    # without requiring the full rhythm circuit.
    _HEAD_CPG_FREQ_HZ: float = 0.6
    # Phase B is the bio-faithful default: rhythm generated by intrinsic
    # voltage-gated oscillation in RMD/SMD, no external sin injection.
    # Set this >0 to ALSO add an external sin (only useful as a debug fallback).
    _HEAD_CPG_AMP: float = 0.0
    # Phase A: inject head CPG at RMD/SMD pacemakers (Mulcahy 2018), letting
    # the rhythm propagate through graded chemical synapses to muscle. With
    # head-ring graded transmission enabled (RMD/SMD/SMB use direct pre.S read
    # via _inject_graded_release), the rhythm reaches the body without spike
    # attenuation. This is the biologically-correct path.
    _HEAD_CPG_AT_RMD: bool = True
    _HEAD_CPG_TARGETS: tuple[str, ...] = ("DB1", "VB1")  # legacy — used when _HEAD_CPG_AT_RMD=False
    _RMD_DORSAL_TARGETS: tuple[str, ...] = ("RMDDL", "RMDDR", "SMDDL", "SMDDR")
    _RMD_VENTRAL_TARGETS: tuple[str, ...] = ("RMDVL", "RMDVR", "SMDVL", "SMDVR")
    _NEURON_TICK_DT: float = 0.002  # seconds/tick (matches NEURAL_TICKS_PER_PHYSICS_STEP=1 × MuJoCo 2ms)

    # ----- Phase B: external intrinsic oscillator on RMD/SMD -----
    # Real RMD/SMD have voltage-gated Ca²⁺ channels producing autonomous rhythmic
    # depolarization (Mulcahy 2018; Carrillo 2010 patch clamp data). Rather than
    # modifying PAULA, we run an external phase oscillator each tick that reads
    # neuron.S (voltage gate) and writes depolarization back to neuron.S.
    # This treats PAULA neurons as thermodynamic state variables that any
    # external dynamic system can manipulate.
    # Phase B default ON: bio-faithful pacemaker model.
    # Baseline must be SMALL relative to leaky decay (~0.05·S per tick at λ=20)
    # so steady-state S settles mid-range rather than saturating at the clamp.
    # Real biology balances NALCN depolarization against shunting conductance.
    # NOTE: dorsal-side RMDs receive less connectome drive than ventral-side
    # ones (the H3 hypothesis test exposed this asymmetry — VB has strong
    # RICR/RIML/AVKL inputs, DB does not). To compensate, dorsal RMDs/SMDs
    # get a slightly larger baseline. Without this asymmetry, ventral
    # connectome bias wins → body locks ventrally.
    _INTRINSIC_OSC_ENABLED: bool = True
    _INTRINSIC_OSC_FREQ_HZ: float = 0.6   # autonomous oscillation frequency
    _INTRINSIC_OSC_AMP: float = 0.025     # Option B: smaller amp keeps S in ±0.5 (no clamp)
    _INTRINSIC_OSC_GATE_S: float = -0.05  # neuron.S threshold above which oscillator is active
    # Option A: baseline=0, amp=0.10 — S oscillates ±2 around 0 (clamped to ±1)
    _INTRINSIC_OSC_BASELINE_D: float = 0.0   # NALCN-like leak on dorsal-side RMD/SMD/SMB
    _INTRINSIC_OSC_BASELINE_V: float = 0.0   # NALCN-like leak on ventral-side RMD/SMD/SMB
    # Backward-compat: keep old single-baseline knob; if user sets it, it's
    # treated as the ventral baseline and dorsal gets a +0.03 bias.
    _INTRINSIC_OSC_BASELINE: float = 0.01
    _intrinsic_osc_phases: dict[str, float] = {}

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

            # For chemosensory neurons, compute delta_c via dual-EMA bandpass
            # filter. Fast EMA tracks head-sweep, slow EMA tracks the
            # environmental gradient. Their difference cancels undulation
            # noise while preserving the true navigational signal.
            # Saturation is applied downstream (tanh soft-cap on the
            # amplified m0/m1) so the motor limit cycle isn't destroyed
            # when the worm reaches high-concentration zones.
            if neuron_name in self._CHEMOSENSORY_NAMES:
                af = self._CHEM_EMA_ALPHA_FAST
                a_s = self._CHEM_EMA_ALPHA_SLOW
                fast = self._chem_ema_fast.get(neuron_name, clamped)
                slow = self._chem_ema_slow.get(neuron_name, clamped)
                fast = af * clamped + (1.0 - af) * fast
                slow = a_s * clamped + (1.0 - a_s) * slow
                self._chem_ema_fast[neuron_name] = fast
                self._chem_ema_slow[neuron_name] = slow
                delta_c = float(fast - slow)
                delta_accum += delta_c
                n_chem += 1
            else:
                prev = self._prev_sensory.get(neuron_name, clamped)
                delta_c = clamped - prev

            if neuron_name in self._OFF_CELL_NAMES:
                # OFF-cell: M1 from absolute concentration level (receptor-
                # mediated suppression), M0 from concentration decrease
                # (burst facilitation on stimulus removal).
                m1 = (
                    float(np.tanh(clamped * self._K_OFF_SUPPRESS / 5.0) * 5.0)
                    if self._enable_m1
                    else 0.0
                )
                if delta_c < -self._STRESS_DEADZONE:
                    excess = abs(delta_c) - self._STRESS_DEADZONE
                    m0 = (
                        float(np.tanh(excess * self._K_STRESS_SYN / 5.0) * 5.0)
                        if self._enable_m0
                        else 0.0
                    )
                else:
                    m0 = 0.0
            elif delta_c < -self._STRESS_DEADZONE:
                excess = abs(delta_c) - self._STRESS_DEADZONE
                m0 = (
                    float(np.tanh(excess * self._K_STRESS_SYN / 5.0) * 5.0)
                    if self._enable_m0
                    else 0.0
                )
                m1 = 0.0
            elif delta_c > 0:
                m0 = 0.0
                m1 = (
                    float(np.tanh(delta_c * self._K_REWARD_SYN / 5.0) * 5.0)
                    if self._enable_m1
                    else 0.0
                )
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

        self._rev_current_tick = int(current_tick)
        self._inject_off_cell_tonic()
        self._inject_motor_proprioception(sensory_inputs)
        self._inject_tonic_forward()
        self._inject_command_noise()
        self._inject_intrinsic_oscillation()
        self._inject_head_cpg(current_tick)

        avg_delta = delta_accum / n_chem if n_chem > 0 else 0.0
        if avg_delta < -self._STRESS_DEADZONE:
            excess = abs(avg_delta) - self._STRESS_DEADZONE
            self._global_m0 = (
                float(np.tanh(excess * self._K_VOL_STRESS / 2.0) * 2.0)
                if self._enable_m0
                else 0.0
            )
            self._global_m1 = 0.0
        elif avg_delta > 0:
            self._global_m0 = 0.0
            self._global_m1 = (
                float(np.tanh(avg_delta * self._K_VOL_REWARD / 2.0) * 2.0)
                if self._enable_m1
                else 0.0
            )
        else:
            self._global_m0 = 0.0
            self._global_m1 = 0.0

    def _inject_tonic_forward(self) -> None:
        """Inject tonic depolarization into forward command interneurons.

        AVB and PVC receive constant tonic drive (NALCN-like) keeping the
        forward circuit "alive" — this is the resting state. Reversal episodes
        are triggered separately (see _inject_reversal_drive) by direct
        depolarization of AVA, not by competing with AVB tonic.

        Removed: the flip-flop sigmoid that previously created a positive
        feedback loop locking the system into AVA-dominant state once any
        connectome activity pushed AVA above AVB.
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

    def _inject_command_noise(self) -> None:
        """Add Ornstein-Uhlenbeck noise to forward + backward command interneurons.

        Real C. elegans run-turn-run switching is stochastic. Without intrinsic
        neural noise the deterministic connectome has a single attractor
        (continuous forward) and the worm never reverses spontaneously. Adding
        OU noise to AVA/AVB/AVD/AVE drives spontaneous AVA depolarisation
        events that trigger biological-style reversals.
        """
        if self._network is None or self._CMD_NOISE_SIGMA <= 0:
            return
        sigma = float(self._CMD_NOISE_SIGMA)
        tau = float(max(1.0, self._CMD_NOISE_TAU))
        decay = 1.0 - 1.0 / tau
        # Pre-factor for OU step (gives steady-state std ≈ sigma).
        kick = sigma * np.sqrt(2.0 / tau)
        for name in (self._FWD_CMD_NAMES | self._BKW_CMD_NAMES):
            nid = self._name_to_id.get(name)
            if nid is None:
                continue
            n = self._network.network.neurons.get(nid)
            if n is None:
                continue
            prev = self._cmd_noise_state.get(name, 0.0)
            new_noise = decay * prev + kick * float(np.random.randn())
            self._cmd_noise_state[name] = new_noise
            n.S += new_noise

    def _inject_head_cpg(self, current_tick: int) -> None:
        """Inject a sinusoidal anti-phase drive into head pacemaker neurons.

        Phase A: by default we target the biological head-ring motor neurons
        (RMDDL/R, RMDVL/R, SMDDL/R, SMDVL/R) — Mulcahy 2018 identifies these
        as the in-vivo CPG. The connectome propagates the rhythm posteriorly
        via natural synapses (RMD → SMB → AVB → B-types).

        Set ``_HEAD_CPG_AT_RMD=False`` to fall back to the legacy DB1/VB1 target
        (the body motor neurons themselves).
        """
        if self._network is None:
            return
        amp = float(self._HEAD_CPG_AMP)
        if amp <= 0.0:
            return
        freq = float(self._HEAD_CPG_FREQ_HZ)
        phase = 2.0 * np.pi * freq * current_tick * self._NEURON_TICK_DT
        drive = amp * float(np.sin(phase))

        if self._HEAD_CPG_AT_RMD:
            # Bio target: RMD/SMD ring motor neurons (real CPG). Spread the
            # drive across the dorsal+ventral pools — connectome propagates.
            for name in self._RMD_DORSAL_TARGETS:
                nid = self._name_to_id.get(name)
                if nid is None: continue
                n = self._network.network.neurons.get(nid)
                if n is not None:
                    n.S += drive  # dorsal RMDs get +drive
            for name in self._RMD_VENTRAL_TARGETS:
                nid = self._name_to_id.get(name)
                if nid is None: continue
                n = self._network.network.neurons.get(nid)
                if n is not None:
                    n.S -= drive  # ventral RMDs get -drive (anti-phase)
            return

        # Legacy: directly inject into body motor neurons.
        for name in self._HEAD_CPG_TARGETS:
            nid = self._name_to_id.get(name)
            if nid is None:
                continue
            n = self._network.network.neurons.get(nid)
            if n is None:
                continue
            prefix = name.rstrip("0123456789")
            sign = 1.0 if prefix == "DB" else -1.0
            n.S += sign * drive

    def _inject_intrinsic_oscillation(self) -> None:
        """Phase B: external voltage-gated phase oscillator on RMD/SMD.

        Treats PAULA neurons as thermodynamic state variables: reads neuron.S,
        runs an external Hopf-style oscillator with voltage gating, writes back.
        No modification to neuron.py needed.

        Mechanism (per oscillator-target neuron):
          1. Tonic depolarization (NALCN-like leak) raises S baseline
          2. Voltage gate: gate = sigmoid(S − threshold) — oscillator only
             "alive" when neuron is sufficiently depolarized
          3. Phase advances at constant rate (intrinsic frequency)
          4. Depolarization injection: amp × gate × sin(phase)

        Result: when the neuron is at rest, oscillator is silent. When tonic
        drive lifts S above threshold, oscillator phase-locks and produces
        rhythmic depolarization. This is the phenomenology of the
        Hodgkin-Huxley Ca²⁺ pacemaker without the full ion channel model.
        """
        if self._network is None or not self._INTRINSIC_OSC_ENABLED:
            return
        omega = 2.0 * np.pi * float(self._INTRINSIC_OSC_FREQ_HZ) * self._NEURON_TICK_DT
        amp = float(self._INTRINSIC_OSC_AMP)
        gate_thr = float(self._INTRINSIC_OSC_GATE_S)
        baseline_d = float(self._INTRINSIC_OSC_BASELINE_D)
        baseline_v = float(self._INTRINSIC_OSC_BASELINE_V)
        for name in self._RMD_DORSAL_TARGETS:
            nid = self._name_to_id.get(name)
            if nid is None: continue
            n = self._network.network.neurons.get(nid)
            if n is None: continue
            n.S += baseline_d
            gate = float(np.tanh(max(0.0, n.S - gate_thr) * 5.0))
            phase = self._intrinsic_osc_phases.get(name)
            if phase is None:
                phase = float(np.random.uniform(0.0, 2.0 * np.pi))
            phase = (phase + omega) % (2.0 * np.pi)
            self._intrinsic_osc_phases[name] = phase
            n.S += amp * gate * float(np.sin(phase))   # +sign dorsal
        for name in self._RMD_VENTRAL_TARGETS:
            nid = self._name_to_id.get(name)
            if nid is None: continue
            n = self._network.network.neurons.get(nid)
            if n is None: continue
            n.S += baseline_v
            gate = float(np.tanh(max(0.0, n.S - gate_thr) * 5.0))
            phase = self._intrinsic_osc_phases.get(name)
            if phase is None:
                phase = float(np.random.uniform(0.0, 2.0 * np.pi))
            phase = (phase + omega) % (2.0 * np.pi)
            self._intrinsic_osc_phases[name] = phase
            n.S -= amp * gate * float(np.sin(phase))   # -sign ventral

    def _inject_off_cell_tonic(self) -> None:
        """Tonic baseline for OFF-cell sensory neurons.

        Models constitutive activity of AWC/ASER at rest — these neurons
        fire tonically in the absence of stimulus (Chalasani et al. 2007).
        """
        if self._network is None:
            return
        for name in self._OFF_CELL_NAMES:
            nid = self._name_to_id.get(name)
            if nid is None:
                continue
            n = self._network.network.neurons.get(nid)
            if n is not None:
                n.S += self._TONIC_OFF_CELL

    def _inject_motor_proprioception(self, sensory_inputs: dict[str, float]) -> None:
        """Proprioceptive drive for B-type and A-type motor neurons.

        Direct S injection models intrinsic mechanosensitivity of motor
        neuron processes (Wen et al. 2012; Fouad 2018). B-types sense the
        anterior joint and propagate forward wave; A-types sense the
        posterior joint and propagate backward wave (sign-flipped, see
        sensors.py:_encode_motor_proprioception).

        Gating: B-type proprio is active when AVB dominates (forward state);
        A-type proprio is active when AVA dominates (backward state). This
        matches biology — both proprio reflexes are intrinsic, but only the
        currently-active motor population experiences depolarization above
        threshold (the inactive class is hyperpolarized by the command
        interneurons). Implementation reuses the same flip-flop gate as
        ``_inject_tonic_forward``.

        Tail-decay applies to B-types (anterior strong, tail weak — wave
        starts at head) and is REVERSED for A-types (tail strong, anterior
        weak — wave starts at tail for backward locomotion).
        """
        if self._network is None:
            return
        # Compute fwd/bkw gates from current command-interneuron state.
        fwd_mean_S = 0.0; n_fwd = 0
        for name in self._FWD_CMD_NAMES:
            nid = self._name_to_id.get(name)
            if nid is None: continue
            nn = self._network.network.neurons.get(nid)
            if nn is not None:
                fwd_mean_S += nn.S; n_fwd += 1
        if n_fwd: fwd_mean_S /= n_fwd
        bkw_mean_S = 0.0; n_bkw = 0
        for name in self._BKW_CMD_NAMES:
            nid = self._name_to_id.get(name)
            if nid is None: continue
            nn = self._network.network.neurons.get(nid)
            if nn is not None:
                bkw_mean_S += nn.S; n_bkw += 1
        if n_bkw: bkw_mean_S /= n_bkw
        # Reversal latch — biological plateau dynamics. The latch is driven by
        # AVA mean S crossing _REV_LATCH_TRIGGER_S. While active, A-type proprio
        # is engaged (bkw_gate = 1) and B-type proprio is attenuated (fwd_gate = 0).
        # Outside the latch, B-type proprio always on (fwd_gate=1), A-type off.
        if self._rev_state == "active":
            if self._rev_current_tick >= self._rev_state_until_tick:
                self._rev_state = "refractory"
                refr = int(np.random.uniform(*self._REV_REFRACTORY_TICKS_RANGE))
                self._rev_state_until_tick = self._rev_current_tick + refr
        elif self._rev_state == "refractory":
            if self._rev_current_tick >= self._rev_state_until_tick:
                self._rev_state = "idle"
        else:  # idle
            if bkw_mean_S > self._REV_LATCH_TRIGGER_S:
                self._rev_state = "active"
                dur = int(np.random.uniform(*self._REV_LATCH_DURATION_TICKS_RANGE))
                self._rev_state_until_tick = self._rev_current_tick + dur

        if self._rev_state == "active":
            fwd_gate = 0.0
            bkw_gate_sharp = 1.0
        else:
            fwd_gate = 1.0
            bkw_gate_sharp = 0.0

        prefix_str = "_mpr_"
        decay = float(self._PROPRIO_TAIL_DECAY)
        for key, val in sensory_inputs.items():
            if not key.startswith(prefix_str):
                continue
            motor_name = key[len(prefix_str) :]
            nid = self._name_to_id.get(motor_name)
            if nid is None:
                continue
            n = self._network.network.neurons.get(nid)
            if n is None:
                continue
            frac = float(MOTOR_NEURON_POSITIONS.get(motor_name, 0.0))
            prefix = motor_name.rstrip("0123456789")
            if prefix in ("DA", "VA"):
                # A-type: tail strong (backward wave starts at tail). Active only
                # when backward command (AVA) dominates — sharp gate so A-types
                # do nothing during forward locomotion.
                gain = self._PROPRIO_MOTOR_GAIN * (1.0 - (1.0 - frac) * decay) * bkw_gate_sharp
            else:
                # B-type: head strong (forward wave starts at head). Always
                # active — B-type proprio is part of the substrate-level
                # locomotion machinery that doesn't depend on the flip-flop.
                gain = self._PROPRIO_MOTOR_GAIN * (1.0 - frac * decay)
            n.S += val * gain

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
    # GABAergic D-type contribution to muscle drive vs cholinergic B/A excitation.
    # Real C. elegans D-types provide reciprocal cross-inhibition roughly comparable
    # in strength to ACh excitation; the previous 0.3 default was tuned for an
    # earlier regime and is the proximal cause of saturation lockup at any joint
    # range, since dorsal/ventral excitation can pin a joint without enough
    # antagonist drive to release it.
    _INHIB_WEIGHT: float = 0.8
    _RECIP_INHIB: float = 0.5
    _FWD_WEIGHT: float = 1.0
    _BKW_WEIGHT: float = 0.3
    # Head ring motor neuron contribution to anterior muscle drive.
    # RMD/SMD/SMB innervate head and anterior body muscles directly in real
    # C. elegans (Mulcahy 2018). Without this path, the RMD pacemaker drive
    # cannot reach muscles via the ventral cord — too many lossy spike-based
    # synapse stages. Low weight (0.5) prevents head over-saturation that
    # otherwise pins the body via proprio runaway.
    _HEAD_RING_WEIGHT: float = 0.5

    # Neuromuscular-junction transfer.
    #   muscle = clip((excit - threshold) * scale, 0, 1)
    # Defaults are pass-through (threshold 0, scale 1).
    _nmj_threshold: float = 0.0
    _nmj_scale: float = 1.0

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

        # Compute head-ring baseline: mean of all RMD/SMD/SMB S values.
        # Used by signed readout for head muscles so dorsal/ventral alternate
        # cleanly even when connectome inputs bias them all positive.
        _head_ring_all = ("RMDDL", "RMDDR", "RMDVL", "RMDVR",
                           "SMDDL", "SMDDR", "SMDVL", "SMDVR",
                           "SMBDL", "SMBDR", "SMBVL", "SMBVR")
        head_ring_S_values = []
        for nm in _head_ring_all:
            nn = self.get_neuron_by_name(nm)
            if nn is not None:
                head_ring_S_values.append(nn.S)
        head_ring_baseline = float(np.mean(head_ring_S_values)) if head_ring_S_values else 0.0

        dorsal_excit = np.zeros(N_BODY_SEGMENTS)
        ventral_excit = np.zeros(N_BODY_SEGMENTS)

        for seg in range(N_BODY_SEGMENTS):
            contributors = self._seg_map[f"seg{seg}"]
            d_exc = 0.0
            v_exc = 0.0
            d_inh = 0.0
            v_inh = 0.0
            for name, w in contributors:
                prefix = name.rstrip("0123456789")
                if prefix == "DB":
                    d_exc += _graded(name) * w * self._FWD_WEIGHT
                elif prefix == "DA":
                    d_exc += _graded(name) * w * self._BKW_WEIGHT
                elif prefix == "AS":
                    d_exc += _graded(name) * w * 0.5
                elif prefix == "VB":
                    v_exc += _graded(name) * w * self._FWD_WEIGHT
                elif prefix == "VA":
                    v_exc += _graded(name) * w * self._BKW_WEIGHT
                elif prefix == "DD":
                    d_inh += _graded(name) * w
                elif prefix == "VD":
                    v_inh += _graded(name) * w
                # Head ring motor neurons (RMD/SMD/SMB) — direct innervation of
                # head/anterior body muscles, real-biology pathway (Mulcahy 2018).
                elif name in ("RMDDL", "RMDDR", "SMDDL", "SMDDR", "SMBDL", "SMBDR"):
                    d_exc += _graded(name) * w * self._HEAD_RING_WEIGHT
                elif name in ("RMDVL", "RMDVR", "SMDVL", "SMDVR", "SMBVL", "SMBVR"):
                    v_exc += _graded(name) * w * self._HEAD_RING_WEIGHT

            dorsal_excit[seg] = d_exc - d_inh * self._INHIB_WEIGHT
            ventral_excit[seg] = v_exc - v_inh * self._INHIB_WEIGHT

        dorsal_excit = np.clip(dorsal_excit, 0.0, 1.0)
        ventral_excit = np.clip(ventral_excit, 0.0, 1.0)

        # Reciprocal inhibition (mechanical antagonism of body-wall muscles)
        d_push = dorsal_excit - ventral_excit * self._RECIP_INHIB
        v_push = ventral_excit - dorsal_excit * self._RECIP_INHIB
        dorsal_excit = np.clip(d_push, 0.0, 1.0)
        ventral_excit = np.clip(v_push, 0.0, 1.0)

        # NMJ transfer: subtract activation threshold then scale.
        nmj_thresh = float(self._nmj_threshold)
        nmj_scale = float(self._nmj_scale)
        if nmj_thresh > 0.0 or nmj_scale != 1.0:
            dorsal_excit = np.clip((dorsal_excit - nmj_thresh) * nmj_scale, 0.0, 1.0)
            ventral_excit = np.clip((ventral_excit - nmj_thresh) * nmj_scale, 0.0, 1.0)

        # LP-filter α: prefer the instance attribute (written by the lab
        # /api/patch → registry wiring) so the live knob actually takes
        # effect; fall back to the module default for scripts that build
        # the nervous system directly.
        alpha = float(
            getattr(self, "_muscle_filter_alpha", MUSCLE_FILTER_ALPHA)
        )
        for seg in range(N_BODY_SEGMENTS):
            for quad in MUSCLE_QUADRANTS:
                key = f"seg{seg}_{quad}"
                target = (
                    float(dorsal_excit[seg])
                    if "D" in quad
                    else float(ventral_excit[seg])
                )
                self._muscle_activations[key] = (
                    alpha * target
                    + (1.0 - alpha) * self._muscle_activations[key]
                )

        return dict(self._muscle_activations)

    def export_live_checkpoint(self) -> dict[str, Any]:
        """Serialise runtime state for live-demo / server restart (not full synapse plasticity)."""
        if self._network is None:
            return {}
        nn = self._network
        neurons_out: dict[str, dict[str, Any]] = {}
        for nid, neuron in nn.network.neurons.items():
            neurons_out[str(nid)] = {
                "S": float(neuron.S),
                "O": float(neuron.O),
                "F_avg": float(neuron.F_avg),
                "t_last_fire": float(neuron.t_last_fire),
                "r": float(neuron.r),
                "b": float(neuron.b),
                "t_ref": float(neuron.t_ref),
                "M_vector": neuron.M_vector.tolist(),
                "pq": [list(row) for row in neuron.propagation_queue],
            }
        return {
            "muscles": dict(self._muscle_activations),
            "prev_sensory": dict(self._prev_sensory),
            "chem_fast": dict(self._chem_ema_fast),
            "chem_slow": dict(self._chem_ema_slow),
            "m0": float(self._global_m0),
            "m1": float(self._global_m1),
            "nn_tick": int(nn.current_tick),
            "neurons": neurons_out,
        }

    def import_live_checkpoint(self, data: dict[str, Any]) -> None:
        """Restore state from :meth:`export_live_checkpoint`. Clears in-flight wheel events."""
        if self._network is None:
            return
        nn = self._network
        for slot in nn.presynaptic_wheel:
            slot.clear()
        for slot in nn.retrograde_wheel:
            slot.clear()
        for k, v in data.get("muscles", {}).items():
            ks = str(k)
            if ks in self._muscle_activations:
                self._muscle_activations[ks] = float(v)
        self._prev_sensory.clear()
        self._prev_sensory.update(
            {str(k): float(v) for k, v in data.get("prev_sensory", {}).items()}
        )
        self._chem_ema_fast.clear()
        self._chem_ema_fast.update(
            {str(k): float(v) for k, v in data.get("chem_fast", {}).items()}
        )
        self._chem_ema_slow.clear()
        self._chem_ema_slow.update(
            {str(k): float(v) for k, v in data.get("chem_slow", {}).items()}
        )
        self._global_m0 = float(data.get("m0", 0.0))
        self._global_m1 = float(data.get("m1", 0.0))
        nn.current_tick = int(data.get("nn_tick", 0))
        for nid_str, ent in data.get("neurons", {}).items():
            nid = int(nid_str)
            neuron = nn.network.neurons.get(nid)
            if neuron is None:
                continue
            neuron.S = float(ent["S"])
            neuron.O = float(ent["O"])
            neuron.F_avg = float(ent["F_avg"])
            neuron.t_last_fire = float(ent["t_last_fire"])
            neuron.r = float(ent["r"])
            neuron.b = float(ent["b"])
            neuron.t_ref = float(ent["t_ref"])
            neuron.M_vector[:] = np.asarray(ent["M_vector"], dtype=float)
            neuron.propagation_queue.clear()
            for row in ent.get("pq", []):
                neuron.propagation_queue.append(
                    (int(row[0]), str(row[1]), float(row[2]), int(row[3]))
                )
            if neuron.propagation_queue:
                heapq.heapify(neuron.propagation_queue)


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


def _off_cell_sensory_params() -> NeuronParameters:
    """OFF-cell sensory neurons (AWC, ASER).

    Low r_base: tonic firing from background + explicit tonic S input.
    Large positive w_r[1]: M1 strongly elevates threshold → suppression.
    Fast gamma[1]: threshold drops quickly on stimulus removal → burst.
    Large lambda: S decays slowly → overlaps with dropping threshold.
    w_tref[1]=0: M1 controls threshold only, not the learning window.
    """
    return NeuronParameters(
        r_base=0.25,
        b_base=0.35,
        c=4,
        lambda_param=20.0,
        p=1.0,
        eta_post=0.002,
        eta_retro=0.001,
        num_neuromodulators=2,
        gamma=np.array([0.98, 0.90]),
        w_r=np.array([-0.2, 1.2]),
        w_b=np.array([-0.2, 1.0]),
        w_tref=np.array([12.0, 0.0]),
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

    off_cell = _off_cell_sensory_params()
    for name in ["AWCL", "AWCR", "ASER"]:
        overrides[name] = off_cell

    return overrides
