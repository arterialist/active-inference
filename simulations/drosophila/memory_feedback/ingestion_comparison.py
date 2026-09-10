"""Add an explicit ingestion predictor using existing PAULA neuron dynamics.

These added cells and projections are engineered, not FlyWire measurements.
No existing KC-to-MBON terminal is reassigned. The predictor starts with zero
context weights and no error-rate amplification. Its independent observation
is accepted nutrient contact, never recurrent PAM activity or a host error.
"""
from copy import deepcopy
import io
import random

import numpy as np

from . import feeding, primary_boundary
from .second_order import ProjectionGate
from ...active_inference.core.runtime_checkpoint import check_buffer_aliases, _serializer
from neuron.neuron import NeuronParameters
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from paula_agent import ckit as k


def serialized(value):
    stream = io.BytesIO()
    _serializer()[1](stream, protocol=5).dump(value)
    return stream.getvalue()


def synapse(cell, sid, weight):
    cell.add_synapse(sid, 1)
    point = cell.postsynaptic_points[sid]
    point.u_i.info = float(weight)
    point.u_i.plast = 0.
    point.u_i.adapt[:] = 0.


def wire(topo, source, target, sid, weight, family):
    cell = topo.neurons[source]
    terminal = next(t for t in range(4095, -1, -1) if t not in cell.distances)
    cell.add_axon_terminal(terminal, 1)
    cell.presynaptic_points[terminal].u_o.mod[:] = 0.
    cell.presynaptic_points[terminal].u_o.info = 1.
    receiver = topo.neurons[target]
    synapse(receiver, sid, weight)
    receiver.register_source(sid, source, terminal)
    topo.connections.append((source, terminal, target, sid))
    return dict(source=source, terminal=terminal, target=target, port=sid,
                birth_weight=float(weight), family=family)


def rebuild(topo):
    topo.num_neurons = len(topo.neurons)
    topo.connection_cache.clear()
    topo.fast_connection_cache.clear()
    for source, terminal, target, sid in topo.connections:
        topo.connection_cache[source, terminal].append((target, sid))
        # Connectome _EmptyTopology's optimizer is a no-op after construction.
        topo.fast_connection_cache[source, terminal].append((topo.neurons[target].input_buffer, sid))
    topo._ext_vec = None


def append_comparison(branch, mapping, manifest):
    topo = branch.network.network
    old_ids = list(topo.neurons)
    before = serialized([topo.neurons[n] for n in old_ids])
    ports = {str(n): topo.neurons[n].params.num_inputs-1 for n in old_ids}
    contexts = [mapping[root] for code in manifest["codes"].values() for root in code]
    assert len(set(contexts)) == len(contexts)
    ids = dict(zip(("observation", "prediction", "positive", "negative"), range(max(old_ids)+1, max(old_ids)+5)))
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    edges = []
    try:
        random.seed(11)
        np.random.seed(11)
        for role, nid in ids.items():
            predictor = role == "prediction"
            node = k.neuron(nid, lam=4 if role in ("observation", "prediction") else 8,
                c=3, r=.6, b=.85, eta_post=1e-5 if predictor else 1e-7,
                eta_retro=1e-7, delta_decay=.99,
                meta=dict(role="ingestion_"+role, graded_gain=1., bounded_plasticity=True,
                          plasticity_rate_boost=0., plasticity_magnitude_decay=.02))
            node["params"]["num_inputs"] = len(contexts)+2 if predictor else 2
            if predictor:
                node["metadata"].update(prediction_ports=list(range(len(contexts))),
                    prediction_error_ports=[[len(contexts), 1], [len(contexts)+1, -1]],
                    prediction_tau_context=8., prediction_tau_error=4.,
                    prediction_cap=1., prediction_boost=0., prediction_half=.01)
            params = node["params"]
            for key in ("gamma", "w_r", "w_b", "w_tref"):
                params[key] = np.asarray(params[key], dtype=float)
            topo.neurons[nid] = PredictiveReceptorNeuron(nid, NeuronParameters(**params),
                log_level="CRITICAL", metadata=node["metadata"])
        observation, prediction, positive, negative = (ids[r] for r in ("observation", "prediction", "positive", "negative"))
        synapse(topo.neurons[observation], 0, 1.)
        synapse(topo.neurons[observation], 1, 0.)
        for sid, source in enumerate(contexts):
            edges.append(wire(topo, source, prediction, sid, 0., "cue_context"))
        for target, sign in ((positive, 1.), (negative, -1.)):
            edges.append(wire(topo, observation, target, 0, sign, "observed"))
            edges.append(wire(topo, prediction, target, 1, -sign, "predicted"))
        edges.append(wire(topo, positive, prediction, len(contexts), 0., "positive_error"))
        edges.append(wire(topo, negative, prediction, len(contexts)+1, 0., "negative_error"))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    # Verify all pre-existing cellular state after removing only new terminals.
    additions = [(topo.neurons[e["source"]], e["terminal"]) for e in edges if e["source"] in old_ids]
    held = [(c, t, c.presynaptic_points.pop(t), c.distances.pop(t)) for c, t in additions]
    exact = serialized([topo.neurons[n] for n in old_ids]) == before
    for cell, terminal, point, distance in held:
        cell.presynaptic_points[terminal] = point
        cell.distances[terminal] = distance
    assert exact
    topo.external_inputs[observation, 0] = dict(info=0., plast=0., mod=np.zeros(2))
    topo.free_synapses.append((observation, 0))
    rebuild(topo)
    check_buffer_aliases(branch.network)
    return dict(ids=ids, contexts=contexts, external_ports=ports, edges=edges,
        original_neuron_ids=old_ids, original_cellular_state_preserved_except_added_terminals=exact,
        action_role="Negative comparison inhibits each existing SMP108 cell through a distinct unit-weight negative dendrite, after comparison verification only.",
        assumptions="Four added engineered cells. Unit observation and signed comparison weights, zero contextual birth weights, existing predictor eta_post=1e-5, prediction_boost=0. Actual ingestion divided by one dose is a physical sensor scale, not an error. All original parameters and terminals preserved at construction.")


def append_action(branch, metadata, targets):
    """Declare the preselected inhibitory path once, after learning verification."""
    topo = branch.network.network
    edges = []
    rng = np.random.get_state()
    try:
        np.random.seed(11)
        for target in targets:
            cell = topo.neurons[target]
            sid = cell.params.num_inputs
            if sid in cell.distances:
                raise ValueError("New action input collides with existing terminal")
            cell.params = deepcopy(cell.params)
            cell.params.num_inputs += 1
            cell.upper_t_ref_bound = cell.params.c*cell.params.num_inputs
            buffer = np.zeros((cell.params.num_inputs, 4), dtype=cell.input_buffer.dtype)
            buffer[:sid] = cell.input_buffer
            cell.input_buffer = buffer
            edges.append(wire(topo, metadata["ids"]["negative"], target, sid, -1., "omission_action"))
    finally:
        np.random.set_state(rng)
    rebuild(topo)
    check_buffer_aliases(branch.network)
    return edges


def action_gate(branch, metadata, targets, cut):
    return ProjectionGate(branch.network, {metadata["ids"]["negative"]}, set(targets), cut)


def step(branch, organism, mapping, manifest, metadata, cue, *, well=False, gate=None):
    net = branch.network
    pump_j, well_j = organism.offer(False, well)
    active = set(manifest["codes"].get(cue, ()))
    def drive(root, value):
        nid = mapping[root]
        net.set_external_input(nid, metadata["external_ports"][str(nid)], value)
    for code in manifest["codes"].values():
        for root in code:
            drive(root, 40. if root in active else 0.)
    for role in primary_boundary.NUTRIENT_ROLES:
        for root in manifest["roles"][role]:
            drive(root, 40.*(pump_j+well_j)/feeding.DOSE_J)
    for role in ("SMP353", "SMP108"):
        for root in manifest["roles"][role]:
            drive(root, 1.4/manifest["factor"] if active else 0.)
    net.set_external_input(metadata["ids"]["observation"], 0, (pump_j+well_j)/feeding.DOSE_J)
    if gate is not None:
        gate.before_step()
    branch.step()
    spike = float(np.mean([net.network.neurons[mapping[r]].O for r in manifest["roles"]["SMP108"]]))
    return organism.step(spike, pump_j, well_j)
