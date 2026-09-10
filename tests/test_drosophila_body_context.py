"""Body context is a nonlinear neural feature, not an added linear cue bias."""
import random

import numpy as np

from simulations.drosophila.memory_feedback import body_context_comparison as body_context
from simulations.drosophila.memory_feedback import ingestion_comparison as wiring
from simulations.active_inference.core.runtime_checkpoint import RuntimeBranch, check_buffer_aliases
from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters


def preparation():
    net = NeuronNetwork(0)
    for nid in range(1, 5):
        cell = Neuron(nid, NeuronParameters(num_inputs=2, eta_post=1e-8, eta_retro=1e-6), log_level="CRITICAL")
        for sid in (0, 1):
            wiring.synapse(cell, sid, 1.)
        cell.add_axon_terminal(100, 1)
        net.network.neurons[nid] = cell
    branch = RuntimeBranch(net, random.getstate(), np.random.get_state(), {})
    metadata = body_context.append_comparison(branch, dict(a=1, b=2, c=3), dict(codes={"A": ["a"], "B": ["b"], "C": ["c"]}))
    return branch, metadata


def test_conjunction_requires_both_actual_drives():
    for cue, position, expected in ((.8, .2, .2), (.8, 0., 0.), (0., .8, 0.), (.8, 2., .8)):
        branch, metadata = preparation()
        gate = branch.network.network.neurons[metadata["body_context"]["gates"][0]]
        gate.input_buffer[:, 0] = [cue, position]
        gate.tick({}, 0)
        assert np.isclose(gate.O, expected)
    predictor = branch.network.network.neurons[metadata["ids"]["prediction"]]
    assert len(predictor.prediction_ports) == len(metadata["contexts"]) == 3
    assert predictor.prediction_boost == 0 and predictor.params.eta_post == 1e-5
    check_buffer_aliases(branch.network)


def test_body_bypass_preserves_cue_drive_instead_of_silencing_prediction():
    branch, metadata = preparation()
    receipt = body_context.bypass_body(branch, metadata)
    assert receipt["all_other_state_and_rng_exact"]
    gate = branch.network.network.neurons[metadata["body_context"]["gates"][0]]
    gate.input_buffer[:, 0] = [.8, 0.]
    gate.tick({}, 0)
    assert np.isclose(gate.O, .8)
    assert gate.synapse_sources[1][0] == metadata["body_context"]["position"]
