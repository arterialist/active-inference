"""Explicit dynamical assumptions over an unchanged FlyWire neuron-pair graph.

The first representation uses ordinary PAULA neurons and a global graded APL.
It cannot reproduce APL's measured spatially local processing. Native forward
and retrograde adaptation remain enabled. No chemical sign is encoded as a
negative release amplitude, which the base PAULA input mask would ignore.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from simulations.paula_loader import ensure_paula_available
from .connectome import Subgraph

ensure_paula_available()
from neuron.neuron import (  # noqa: E402
    Neuron, NeuronParameters, PostsynapticPoint, PostsynapticInputVector,
    PresynapticPoint, PresynapticOutputVector,
)
from neuron.extensions.graded import GradedNeuron  # noqa: E402
from simulations.connectome_loader import _assemble_network  # noqa: E402

PORT_CAPACITY = 2**12


@dataclass(frozen=True)
class Dynamics:
    """Engineering defaults for an execution probe, not fitted fly physiology.

    All classes share thresholds, integration and per-count strength here.
    APL alone differs in release type. One PAULA tick has no assigned physical
    duration yet. The network adds one cleft tick to dendritic delay.
    """
    weight_per_count: float = 0.02
    dendritic_delay_ticks: int = 2
    lambda_ticks: float = 20.0
    cooldown_ticks: int = 3
    threshold: float = 1.0
    cooldown_threshold: float = 1.2
    signal_decay: float = 0.95
    eta_post: float = 1e-8
    # 1e-8 produced no terminal changes in the first execution probe: native
    # float32 error updates rounded away at unit coefficients. Still uncalibrated.
    eta_retro: float = 1e-6
    apl_representation: str = "global_graded"
    apl_graded_gain: float = 0.01
    apl_release_max: float = 1.0

    def validate(self):
        for name in ("weight_per_count", "lambda_ticks", "threshold", "cooldown_threshold",
                     "eta_post", "eta_retro", "apl_graded_gain", "apl_release_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lambda_ticks < 1 or not 0 < self.signal_decay <= 1:
            raise ValueError("Unstable integration/invalid signal decay")
        for name, minimum in (("dendritic_delay_ticks", 0), ("cooldown_ticks", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.apl_representation not in {"global_graded", "spiking_null"}:
            raise ValueError("Unknown APL representation")


@dataclass
class Preparation:
    network: object
    root_to_id: dict[str, int]
    drive_ports: dict[str, int]
    edge_bindings: np.ndarray  # source_row, pre_id, terminal, post_id, synapse
    incoming_boundary_ports: np.ndarray  # source_row, outside_pre_id, post_id, synapse
    outgoing_boundary_terminals: np.ndarray  # source_row, pre_id, terminal, outside_post_id
    assumptions: dict

    def stimulate(self, root: str, info: float):
        """Dedicated experimental input; never an inferred boundary spike train."""
        if root not in self.drive_ports:
            raise ValueError(f"Neuron is outside this preparation: {root}")
        if not math.isfinite(info) or info < 0:
            raise ValueError("Input release must be finite and non-negative")
        self.network.set_external_input(self.root_to_id[root], self.drive_ports[root], info)


def build_paula(graph: Subgraph, dynamics: Dynamics = Dynamics()) -> Preparation:
    graph.validate()
    dynamics.validate()
    pre_inside, post_inside = graph.membership()
    # A cut must not change the native num_inputs-dependent plasticity bounds.
    # Retain ports for EVERY incident pair, even when the other neuron is absent.
    order = np.argsort(graph.edges[:, 8])
    edges = graph.edges[order]
    pre_inside, post_inside = pre_inside[order], post_inside[order]
    incoming = Counter(int(i) for i in edges[post_inside, 3])
    outgoing = Counter(int(i) for i in edges[pre_inside, 2])
    root_to_id = {r: graph.nodes[r]["global_index"] for r in graph.selected}
    for root, nid in root_to_id.items():
        if incoming[nid] + 1 > PORT_CAPACITY or outgoing[nid] > PORT_CAPACITY:
            raise ValueError(f"PAULA port overflow for {root}: {incoming[nid]} incoming + drive, {outgoing[nid]} outgoing; no edges dropped")
    weights = edges[:, 6] * dynamics.weight_per_count
    if not np.isfinite(weights).all() or np.any(np.abs(weights) > 100):
        raise ValueError("Count-to-weight conversion exceeds PAULA weight bounds; no clipping applied")

    neurons = {}
    drive_ports = {}
    for root, nid in root_to_id.items():
        annotation = graph.nodes[root]["annotation"]
        metadata = {"flywire_root_id": root, "annotation": annotation.copy(),
                    "identity_namespace": "FlyWire783/Shiu_global_index"}
        graded = annotation["hemibrain_type"] == "APL" and dynamics.apl_representation == "global_graded"
        if graded:
            metadata.update(graded_gain=dynamics.apl_graded_gain, graded_S0=0.0,
                            graded_max=dynamics.apl_release_max)
        params = NeuronParameters(
            num_inputs=incoming[nid] + 1, lambda_param=dynamics.lambda_ticks,
            c=dynamics.cooldown_ticks, r_base=dynamics.threshold,
            b_base=dynamics.cooldown_threshold, delta_decay=dynamics.signal_decay,
            eta_post=dynamics.eta_post, eta_retro=dynamics.eta_retro,
        )
        neuron_class = GradedNeuron if graded else Neuron
        cell = neuron_class(nid, params, log_level="CRITICAL", metadata=metadata)
        # Construct explicit coefficients without the helpers' random defaults.
        # Terminal distances are unused by the base release equations. Do not
        # write them into the shared `distances` dict and overwrite input delays.
        for sid in range(params.num_inputs):
            cell.postsynaptic_points[sid] = PostsynapticPoint(
                PostsynapticInputVector(info=1.0, plast=0.0, adapt=np.zeros(2)))
            cell.distances[sid] = dynamics.dendritic_delay_ticks
        drive_ports[root] = incoming[nid]
        cell.distances[drive_ports[root]] = 0  # explicit experimental current site
        neurons[nid] = cell

    next_input, next_output = Counter(), Counter()
    connections = []
    bindings = []
    incoming_boundary, outgoing_boundary = [], []
    for edge, weight, has_pre, has_post in zip(edges, weights, pre_inside, post_inside, strict=True):
        pre, post = int(edge[2]), int(edge[3])
        if has_pre:
            terminal = next_output[pre]
            next_output[pre] += 1
            neurons[pre].presynaptic_points[terminal] = PresynapticPoint(
                PresynapticOutputVector(info=1.0, mod=np.zeros(2)), u_i_retro=1.0)
        if has_post:
            synapse = next_input[post]
            next_input[post] += 1
            neurons[post].postsynaptic_points[synapse].u_i.info = float(weight)
        if has_pre and has_post:
            neurons[post].register_source(synapse, pre, terminal)
            connection = (pre, terminal, post, synapse)
            connections.append(connection)
            bindings.append((int(edge[8]), *connection))
        elif has_post:
            incoming_boundary.append((int(edge[8]), pre, post, synapse))
        else:
            outgoing_boundary.append((int(edge[8]), pre, terminal, post))
    network = _assemble_network(neurons, connections)
    network.record_history = False
    actual_cache_edges = sum(map(len, network.network.fast_connection_cache.values()))
    expected_edges = int((pre_inside & post_inside).sum())
    if len(connections) != expected_edges or actual_cache_edges != expected_edges:
        raise RuntimeError("Assembly lost or duplicated anatomical connections")
    for root, sid in drive_ports.items():
        nid = root_to_id[root]
        network.network.free_synapses.append((nid, sid))
        network.set_external_input(nid, sid, 0.0)
    return Preparation(network, root_to_id, drive_ports,
                       np.asarray(bindings, dtype=np.int64).reshape(-1, 5),
                       np.asarray(incoming_boundary, dtype=np.int64).reshape(-1, 4),
                       np.asarray(outgoing_boundary, dtype=np.int64).reshape(-1, 4), {
        "parameters": asdict(dynamics), "physical_seconds_per_tick": None,
        "cleft_delay_ticks": 1,
        "boundary_condition": "outside neurons absent; incoming boundary ports undriven, outgoing boundary terminals disconnected; all incident ports and native input-count-dependent bounds retained",
        "count_conversion": "source_model_sign * synapse_count * weight_per_count on postsynaptic info only",
        "release": "initial positive unit terminal coefficient; one distinct terminal per directed pair; native retrograde adaptation active",
        "plasticity": "native legacy_multiplicative postsynaptic and native retrograde, weak positive rates; no claim of biological calibration or sign invariance",
        "apl_limit": "global graded approximation, lacks measured local integration; inherited timing plasticity is not a non-spiking cellular learning model",
        "spatial_limit": "pair counts only, no contact positions or axonal propagation model",
        "experimental_input": "one additional unit-weight zero-delay input per selected neuron, not claimed as anatomy",
    })
