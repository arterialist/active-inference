"""Existing cue/position coincidence features for the ingestion predictor.

The body's existing positive joint-position afferent is angle/.05 radians.
One existing ConjunctiveGradedNeuron per original KC context port releases
min(KC dendritic drive, position dendritic drive). Predictor weights learn
from these joint features with their existing equation and rate. No linear
body bias, host approach/contact flag, well threshold, grace timer or added
gain is used. The coincidence features and their scale are engineered, not
learned anatomy or a learned sharp contact threshold.
"""
import random

import numpy as np

from . import ingestion_comparison as original
from .ingestion_comparison import append_action, action_gate, serialized
from ...active_inference.components.body.loaded_hinge import afferents
from ...active_inference.core.runtime_checkpoint import check_buffer_aliases
from neuron.neuron import NeuronParameters
from neuron.extensions.conjunctive import ConjunctiveGradedNeuron
from neuron.extensions.experimental.predictive_receptor import PredictiveReceptorNeuron
from paula_agent import ckit as k


def new_cell(nid, cls, metadata):
    node = k.neuron(nid, lam=4, c=3, r=.6, b=.85, eta_post=1e-7, eta_retro=1e-7,
        delta_decay=.99, meta=metadata)
    params = node["params"]
    params["num_inputs"] = 2
    for key in ("gamma", "w_r", "w_b", "w_tref"):
        params[key] = np.asarray(params[key], dtype=float)
    return cls(nid, NeuronParameters(**params), log_level="CRITICAL", metadata=metadata)


def append_comparison(branch, mapping, manifest):
    metadata = original.append_comparison(branch, mapping, manifest)
    topo = branch.network.network
    position = max(topo.neurons)+1
    gates = list(range(position+1, position+1+len(metadata["contexts"])))
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    edges = []
    try:
        random.seed(11)
        np.random.seed(11)
        topo.neurons[position] = new_cell(position, PredictiveReceptorNeuron,
            dict(role="physical_joint_position", graded_gain=1., bounded_plasticity=True, plasticity_rate_boost=0.))
        original.synapse(topo.neurons[position], 0, 1.)
        original.synapse(topo.neurons[position], 1, 0.)
        contextual = [e for e in metadata["edges"] if e["family"] == "cue_context"]
        assert len(contextual) == len(gates)
        for gate, edge in zip(gates, contextual, strict=True):
            topo.neurons[gate] = new_cell(gate, ConjunctiveGradedNeuron,
                dict(role="cue_position_coincidence", graded_gain=1., conjunctive_graded_gain=1.,
                     conjunctive_ring_synapse=0, conjunctive_velocity_synapse=1,
                     conjunctive_ring_tau=1., conjunctive_velocity_tau=1.))
            # Reuse the dedicated KC terminal; no anatomical old target changes.
            old = (edge["source"], edge["terminal"], edge["target"], edge["port"])
            topo.connections.remove(old)
            original.synapse(topo.neurons[gate], 0, 1.)
            topo.neurons[gate].register_source(0, edge["source"], edge["terminal"])
            topo.connections.append((edge["source"], edge["terminal"], gate, 0))
            edges.append(dict(source=edge["source"], terminal=edge["terminal"], target=gate, port=0, birth_weight=1., family="cue_to_coincidence"))
            edges.append(original.wire(topo, position, gate, 1, 1., "position_to_coincidence"))
            edges.append(original.wire(topo, gate, edge["target"], edge["port"], 0., "conjunctive_prediction_context"))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    topo.external_inputs[position, 0] = dict(info=0., plast=0., mod=np.zeros(2))
    topo.free_synapses.append((position, 0))
    original.rebuild(topo)
    check_buffer_aliases(branch.network)
    metadata["edges"] = [e for e in metadata["edges"] if e["family"] != "cue_context"]+edges
    metadata["body_context"] = dict(position=position, gates=gates, afferent_index=2,
        physical_scale="Existing loaded_hinge.afferents positive position = max(0, angle/.05)",
        feature="One min(actual KC drive, actual positive-position drive) coincidence cell per existing predictor input. Original predictor capacity, zero initial weights, eta_post and zero boost retained.",
        learning="The joint features are fixed; predictor weights learn nutrient expectation from them. This is nonlinear cue/body interaction, not a learned sharp contact boundary.",
        control="At the same learned state, substitute the cue dendrite for the body dendrite only in the coincidence readout. min(cue,cue) preserves cue drive and removes body dependence. All physical afferents, connections and native adaptation remain.",
        limits="193 added engineered cells. Existing ConjunctiveGradedNeuron is used as a generic coincidence component; no claim that these are measured fly neurons. Positive position alone omits velocity, force and learned spatial boundaries.")
    return metadata


def bypass_body(branch, metadata):
    cells = [branch.network.network.neurons[n] for n in metadata["body_context"]["gates"]]
    before = serialized(branch)
    original_ports = [cell._cg_velocity_synapses for cell in cells]
    assert all(ports == (1,) for ports in original_ports)
    for cell in cells:
        cell._cg_velocity_synapses = (0,)
    for cell, ports in zip(cells, original_ports, strict=True):
        cell._cg_velocity_synapses = ports
    assert serialized(branch) == before
    for cell in cells:
        cell._cg_velocity_synapses = (0,)
    return dict(changed_readouts=len(cells), all_other_state_and_rng_exact=True,
        change="Coincidence second argument reads existing cue dendrite instead of body dendrite. No gain change, weight reset, missing cue, or per-tick host decision.")


def step(branch, organism, mapping, manifest, metadata, cue, *, well=False, gate=None):
    # Sample the real physical afferent before the next neural/body step.
    value = float(afferents(organism.body)[metadata["body_context"]["afferent_index"]])
    branch.network.set_external_input(metadata["body_context"]["position"], 0, value)
    return original.step(branch, organism, mapping, manifest, metadata, cue, well=well, gate=gate)
