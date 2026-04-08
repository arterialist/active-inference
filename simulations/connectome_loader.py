"""
Generic connectome -> PAULA NeuronNetwork builder.

Parses a ConnectomeData object and constructs a NeuronNetwork whose
topology exactly mirrors the biological wiring.

Key design choices
------------------
- Each biological neuron becomes one PAULA Neuron instance.
- Chemical synapses become directed connections; synapse count (weight)
  scales the initial u_i.info value.
- Gap junctions become two anti-parallel directed connections so PAULA's
  retrograde signaling can flow both ways.
- The number of PAULA postsynaptic_points per neuron is set to the
  in-degree of that neuron in the connectome (cap at PAULA_SYNAPSE_LIMIT).
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from simulations.paula_loader import ensure_paula_available

ensure_paula_available()
from neuron.neuron import Neuron, NeuronParameters  # noqa: E402
from neuron.network import NeuronNetwork  # noqa: E402

# PAULA model limits (12-bit synapse/terminal IDs)
PAULA_SYNAPSE_LIMIT = 4095
CHEMICAL_SYNAPSE_DISTANCE = 2
GAP_JUNCTION_DISTANCE = 1


# -----------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------


@dataclass
class NeuronInfo:
    """Metadata for a single biological neuron."""

    name: str
    neuron_type: str          # 'sensory' | 'interneuron' | 'motor' | 'unknown'
    paula_id: int             # integer ID used by PAULA Neuron
    in_degree_chem: int = 0
    in_degree_gap: int = 0
    out_degree_chem: int = 0
    out_degree_gap: int = 0


@dataclass
class SynapticEdge:
    """A single directed connection between two neurons."""

    pre_name: str
    post_name: str
    synapse_type: str   # 'chemical' | 'gap_junction'
    weight: float       # raw synapse count from connectome


@dataclass
class ConnectomeData:
    """
    Parsed connectome: neuron list + connection lists.

    Produced by organism-specific loaders (e.g. c_elegans/connectome.py)
    and consumed by build_paula_network().
    """

    neurons: list[NeuronInfo]
    chemical_edges: list[SynapticEdge]
    gap_junction_edges: list[SynapticEdge]
    name_to_info: dict[str, NeuronInfo] = field(init=False)

    def __post_init__(self) -> None:
        self.name_to_info = {n.name: n for n in self.neurons}

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def all_edges(self) -> list[SynapticEdge]:
        return self.chemical_edges + self.gap_junction_edges


# -----------------------------------------------------------------------
# PAULA network builder
# -----------------------------------------------------------------------


def build_paula_network(
    connectome: ConnectomeData,
    base_params: NeuronParameters | None = None,
    sensory_params: NeuronParameters | None = None,
    motor_params: NeuronParameters | None = None,
    interneuron_params: NeuronParameters | None = None,
    weight_max: float = 5.0,
    log_level: str = "WARNING",
    param_overrides: dict[str, NeuronParameters] | None = None,
) -> tuple[NeuronNetwork, dict[str, int]]:
    """
    Build a PAULA NeuronNetwork from a ConnectomeData object.

    Args:
        connectome:          Parsed connectome.
        base_params:         Default NeuronParameters for all neurons.
        sensory_params:      Override params for sensory neurons.
        motor_params:        Override params for motor neurons.
        interneuron_params:  Override params for interneurons.
        weight_max:          The heaviest biological synapse maps to this
                             PAULA u_i.info value.  Lighter synapses scale
                             proportionally, preserving the natural ratio.
        log_level:           Log verbosity for individual PAULA neurons.
        param_overrides:     Per-neuron-name parameter overrides (highest
                             priority, used for sub-class specialisation
                             like GABAergic motor neurons or command
                             interneurons).

    Returns:
        Tuple of:
          - NeuronNetwork ready to simulate
          - name_to_paula_id dict for interfacing sensors/motors
    """
    if base_params is None:
        base_params = _default_base_params()
    if sensory_params is None:
        sensory_params = _default_sensory_params()
    if motor_params is None:
        motor_params = _default_motor_params()
    if interneuron_params is None:
        interneuron_params = _default_interneuron_params()

    logger.info(
        f"Building PAULA network from connectome: "
        f"{connectome.n_neurons} neurons, "
        f"{len(connectome.chemical_edges)} chemical synapses, "
        f"{len(connectome.gap_junction_edges)} gap junctions"
    )

    # Derive proportional weight_scale from the heaviest *chemical* synapse
    # so the ratio is preserved.  Chemical and gap-junction weights are
    # scaled independently: they have different biophysics.
    chem_weights = [e.weight for e in connectome.chemical_edges if e.weight > 0]
    gap_weights = [e.weight for e in connectome.gap_junction_edges if e.weight > 0]
    max_chem = max(chem_weights) if chem_weights else 1.0
    max_gap = max(gap_weights) if gap_weights else 1.0
    weight_scale = weight_max / max_chem
    gap_scale = weight_max / max_gap
    logger.info(
        f"Weight scaling: max chem={max_chem}, max gap={max_gap} → "
        f"PAULA weight_max={weight_max:.2f}  "
        f"(chem_scale={weight_scale:.4f}, gap_scale={gap_scale:.4f})"
    )

    # ---- 1. Assign PAULA IDs and compute degree counts ----------------
    _assign_degrees(connectome)

    name_to_id: dict[str, int] = {
        n.name: n.paula_id for n in connectome.neurons
    }

    # ---- 2. Create Neuron instances -----------------------------------
    neurons: dict[int, Neuron] = {}

    if param_overrides is None:
        param_overrides = {}

    for info in connectome.neurons:
        if info.name in param_overrides:
            params = param_overrides[info.name]
        else:
            params = _select_params(
                info.neuron_type,
                base_params,
                sensory_params,
                motor_params,
                interneuron_params,
            )
        # num_inputs must match the actual number of postsynaptic_points
        # we will add below.  Cap at PAULA's synapse limit.
        in_degree = min(
            info.in_degree_chem + info.in_degree_gap, PAULA_SYNAPSE_LIMIT
        )
        # Always give at least 1 input slot so the neuron can receive background
        in_degree = max(in_degree, 1)

        params_copy = _copy_params_with_inputs(params, num_inputs=in_degree)

        neuron = Neuron(
            neuron_id=info.paula_id,
            params=params_copy,
            log_level=log_level,
            metadata={"name": info.name, "type": info.neuron_type},
        )
        neurons[info.paula_id] = neuron

    # ---- 3. Add synaptic points and connections -----------------------
    # We need to assign synapse slot indices (0 .. in_degree-1) per neuron.
    next_synapse_slot: dict[int, int] = {n.paula_id: 0 for n in connectome.neurons}
    next_terminal_slot: dict[int, int] = {n.paula_id: 0 for n in connectome.neurons}

    # Collect connections list: (pre_paula_id, pre_terminal, post_paula_id, post_synapse)
    connection_list: list[tuple[int, int, int, int]] = []

    def _add_edge(
        pre_name: str,
        post_name: str,
        weight: float,
        synapse_type: str,
        scale: float,
    ) -> None:
        if pre_name not in name_to_id or post_name not in name_to_id:
            return
        pre_id = name_to_id[pre_name]
        post_id = name_to_id[post_name]

        pre_neuron = neurons[pre_id]
        post_neuron = neurons[post_id]

        t_slot = next_terminal_slot[pre_id]
        if t_slot >= PAULA_SYNAPSE_LIMIT:
            return
        next_terminal_slot[pre_id] += 1

        s_slot = next_synapse_slot[post_id]
        if s_slot >= post_neuron.params.num_inputs:
            return
        next_synapse_slot[post_id] += 1

        distance = (GAP_JUNCTION_DISTANCE if synapse_type == "gap_junction"
                    else CHEMICAL_SYNAPSE_DISTANCE)

        pre_neuron.add_axon_terminal(
            terminal_id=t_slot, distance_from_hillock=distance
        )
        post_neuron.add_synapse(
            synapse_id=s_slot, distance_to_hillock=distance
        )
        scaled_weight = max(float(weight * scale), 0.01)
        post_neuron.postsynaptic_points[s_slot].u_i.info = scaled_weight
        # Deterministic baseline plasticity.  PAULA's V_local =
        # info_val * (u_i.info + u_i.plast), so plast = 1.0 ensures
        # even weak synapses conduct while preserving the weight ratio.
        post_neuron.postsynaptic_points[s_slot].u_i.plast = 1.0

        post_neuron.register_source(
            synapse_id=s_slot,
            source_neuron_id=pre_id,
            source_terminal_id=t_slot,
        )
        pre_neuron.presynaptic_points[t_slot].u_o.info = scaled_weight

        connection_list.append((pre_id, t_slot, post_id, s_slot))

    for edge in connectome.chemical_edges:
        _add_edge(edge.pre_name, edge.post_name, edge.weight, "chemical",
                  scale=weight_scale)

    for edge in connectome.gap_junction_edges:
        _add_edge(edge.pre_name, edge.post_name, edge.weight, "gap_junction",
                  scale=gap_scale)
        _add_edge(edge.post_name, edge.pre_name, edge.weight, "gap_junction",
                  scale=gap_scale)

    # ---- 4. Assemble NeuronNetwork ------------------------------------
    network = _assemble_network(neurons, connection_list)

    n_connections = len(connection_list)
    logger.info(
        f"PAULA network assembled: {len(neurons)} neurons, "
        f"{n_connections} PAULA connections"
    )

    return network, name_to_id


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------


def _assign_degrees(connectome: ConnectomeData) -> None:
    """Compute in/out degrees and assign stable integer PAULA IDs.

    Idempotent: resets all degree counters before recomputing so that
    calling this multiple times (e.g. on successive resets) gives the
    same result.
    """
    for idx, info in enumerate(connectome.neurons):
        info.paula_id = idx  # stable 0-based index
        # Reset counters so repeated calls don't accumulate
        info.in_degree_chem = 0
        info.out_degree_chem = 0
        info.in_degree_gap = 0
        info.out_degree_gap = 0

    for edge in connectome.chemical_edges:
        pre = connectome.name_to_info.get(edge.pre_name)
        post = connectome.name_to_info.get(edge.post_name)
        if pre:
            pre.out_degree_chem += 1
        if post:
            post.in_degree_chem += 1

    for edge in connectome.gap_junction_edges:
        pre = connectome.name_to_info.get(edge.pre_name)
        post = connectome.name_to_info.get(edge.post_name)
        if pre:
            pre.out_degree_gap += 1
            pre.in_degree_gap += 1  # gap junctions are bidirectional
        if post:
            post.out_degree_gap += 1
            post.in_degree_gap += 1


def _select_params(
    neuron_type: str,
    base: NeuronParameters,
    sensory: NeuronParameters,
    motor: NeuronParameters,
    interneuron: NeuronParameters,
) -> NeuronParameters:
    """Select NeuronParameters by neuron type. Falls back to base for unknown types."""
    param_map = {
        "sensory": sensory,
        "motor": motor,
        "interneuron": interneuron,
    }
    return param_map.get(neuron_type, base)


def _copy_params_with_inputs(
    params: NeuronParameters, num_inputs: int
) -> NeuronParameters:
    """Return a copy of params with num_inputs set appropriately."""
    return NeuronParameters(
        delta_decay=params.delta_decay,
        beta_avg=params.beta_avg,
        eta_post=params.eta_post,
        eta_retro=params.eta_retro,
        c=params.c,
        lambda_param=params.lambda_param,
        p=params.p,
        r_base=params.r_base,
        b_base=params.b_base,
        gamma=params.gamma.copy(),
        w_r=params.w_r.copy(),
        w_b=params.w_b.copy(),
        w_tref=params.w_tref.copy(),
        num_neuromodulators=params.num_neuromodulators,
        num_inputs=num_inputs,
    )


def _assemble_network(
    neurons: dict[int, Neuron],
    connections: list[tuple[int, int, int, int]],
) -> NeuronNetwork:
    """
    Build a NeuronNetwork from pre-created Neuron objects and connections.

    We bypass NetworkTopology._create_network() (which creates random
    connections) by directly injecting the neurons and wiring.
    """

    # Build a minimal topology container that duck-types NetworkTopology
    topology = _EmptyTopology(neurons, connections)

    # Construct NeuronNetwork without calling __init__ (which would create random topology)
    network = NeuronNetwork.__new__(NeuronNetwork)
    network.network = topology          # NeuronNetwork uses self.network internally
    network.current_tick = 0
    network.max_history = 1000

    # Calendar wheel for signal scheduling (matches NeuronNetwork.__init__)
    network.max_delay = 10
    network.wheel_size = network.max_delay + 1
    network.presynaptic_wheel = [[] for _ in range(network.wheel_size)]
    network.retrograde_wheel = [[] for _ in range(network.wheel_size)]

    # History tracking (matches NeuronNetwork.__init__)
    network.history = {
        "ticks": deque(maxlen=network.max_history),
        "neuron_states": defaultdict(
            lambda: {
                "membrane_potential": deque(maxlen=1000),
                "firing": deque(maxlen=1000),
                "firing_rate": deque(maxlen=1000),
                "output": deque(maxlen=1000),
            }
        ),
        "network_activity": deque(maxlen=network.max_history),
    }

    # Default off (matches NeuronNetwork.__init__). Set PAULA_RECORD_HISTORY=1 to
    # benchmark or debug with per-tick PAULA deques enabled.
    network.record_history = os.environ.get("PAULA_RECORD_HISTORY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    return network


class _EmptyTopology:
    """Minimal duck-type of NetworkTopology backed by pre-built neurons."""

    def __init__(
        self,
        neurons: dict[int, Neuron],
        connections: list[tuple[int, int, int, int]],
    ):
        self.neurons = neurons
        self.connections = connections
        self.num_neurons = len(neurons)

        # Build connection cache: (pre_id, terminal_id) -> [(post_id, synapse_id)]
        from collections import defaultdict
        self.connection_cache: dict[tuple, list] = defaultdict(list)
        self.fast_connection_cache: dict[tuple, list] = defaultdict(list)
        self.free_synapses: list = []
        self.external_inputs: dict = {}

        for pre_id, terminal_id, post_id, synapse_id in connections:
            self.connection_cache[(pre_id, terminal_id)].append(
                (post_id, synapse_id)
            )

        # Build fast cache (direct numpy buffer references)
        for pre_id, terminal_id, post_id, synapse_id in connections:
            post_neuron = neurons[post_id]
            if synapse_id < post_neuron.params.num_inputs:
                buf_ref = post_neuron.input_buffer
                self.fast_connection_cache[(pre_id, terminal_id)].append(
                    (buf_ref, synapse_id)
                )

    def set_external_input(
        self, input_key: tuple[int, int], info: float, mod: np.ndarray | None = None
    ) -> None:
        """Set external input for a specific synapse (duck-type of NetworkTopology)."""
        if input_key not in self.external_inputs:
            self.external_inputs[input_key] = {
                "info": 0.0,
                "mod": np.zeros(2),
                "plast": 0.0,
            }
        self.external_inputs[input_key]["info"] = info
        if mod is not None:
            self.external_inputs[input_key]["mod"] = mod
        if "plast" not in self.external_inputs[input_key]:
            self.external_inputs[input_key]["plast"] = 0.0

    def optimize_runtime_connections(self) -> None:
        pass  # Already done in __init__


# -----------------------------------------------------------------------
# Default parameter presets
# -----------------------------------------------------------------------


def _default_base_params() -> NeuronParameters:
    return NeuronParameters(
        r_base=0.9,
        b_base=1.1,
        c=8,
        lambda_param=15.0,
        p=1.0,
        eta_post=0.005,
        eta_retro=0.005,
        num_neuromodulators=2,
    )


def _default_sensory_params() -> NeuronParameters:
    """
    Sensory neurons have lower thresholds (more responsive to input)
    and faster time constants.
    """
    return NeuronParameters(
        r_base=0.6,
        b_base=0.8,
        c=5,
        lambda_param=8.0,
        p=1.0,
        eta_post=0.01,
        eta_retro=0.005,
        num_neuromodulators=2,
    )


def _default_motor_params() -> NeuronParameters:
    """
    Motor neurons integrate strongly and have moderate thresholds.
    Faster learning rates because motor control needs rapid adaptation.
    """
    return NeuronParameters(
        r_base=0.8,
        b_base=1.0,
        c=6,
        lambda_param=12.0,
        p=1.2,
        eta_post=0.01,
        eta_retro=0.01,
        num_neuromodulators=2,
    )


def _default_interneuron_params() -> NeuronParameters:
    """
    Interneurons have the highest thresholds and longest time constants —
    they act as integrators / pattern generators.
    """
    return NeuronParameters(
        r_base=1.0,
        b_base=1.3,
        c=10,
        lambda_param=20.0,
        p=1.0,
        eta_post=0.005,
        eta_retro=0.005,
        num_neuromodulators=2,
    )
