"""Additive wiring must preserve memory and isolate the action-pathway cut."""
import random
from types import SimpleNamespace

import numpy as np
import pytest

from simulations.drosophila.memory_feedback import ingestion_comparison as composition
from simulations.active_inference.core.runtime_checkpoint import RuntimeBranch, check_buffer_aliases
from neuron.network import NeuronNetwork
from neuron.neuron import Neuron, NeuronParameters, RetrogradeSignalEvent
from simulations.drosophila.memory_feedback.terminal_credit import TerminalCreditNeuron


def preparation():
    net = NeuronNetwork(0)
    for nid in range(1, 5):
        cell = Neuron(nid, NeuronParameters(num_inputs=2, eta_post=1e-8, eta_retro=1e-6), log_level="CRITICAL")
        for sid in (0, 1):
            composition.synapse(cell, sid, 1.)
        cell.add_axon_terminal(100, 1)
        cell.presynaptic_points[100].u_o.info = .37
        net.network.neurons[nid] = cell
    branch = RuntimeBranch(net, random.getstate(), np.random.get_state(), {})
    manifest = dict(codes={"A": ["a"], "B": ["b"], "C": ["c"]})
    return branch, dict(a=1, b=2, c=3), manifest


def test_addition_preserves_existing_state_and_rebuilds_direct_buffers():
    branch, mapping, manifest = preparation()
    # The measured-connectome topology's optimizer is intentionally a no-op.
    branch.network.network.optimize_runtime_connections = lambda: None
    metadata = composition.append_comparison(branch, mapping, manifest)
    assert metadata["original_cellular_state_preserved_except_added_terminals"]
    assert all(branch.network.network.neurons[n].presynaptic_points[100].u_o.info == .37 for n in range(1, 5))
    predictor = branch.network.network.neurons[metadata["ids"]["prediction"]]
    assert predictor.prediction_boost == 0
    assert all(predictor.postsynaptic_points[s].u_i.info == 0 for s in predictor.prediction_ports)
    assert predictor.params.eta_post > 0 and predictor.params.eta_retro > 0
    check_buffer_aliases(branch.network)


def test_action_cut_preserves_error_feedback_and_old_cue_port():
    branch, mapping, manifest = preparation()
    metadata = composition.append_comparison(branch, mapping, manifest)
    target = branch.network.network.neurons[4]
    old_buffer = target.input_buffer.copy()
    old_port = metadata["external_ports"]["4"]
    action = composition.append_action(branch, metadata, [4])[0]
    assert action["birth_weight"] == -1.
    assert old_port == 1 and action["port"] == 2
    assert np.array_equal(target.input_buffer[:2], old_buffer)
    gate = composition.action_gate(branch, metadata, [4], True)
    negative = metadata["ids"]["negative"]
    teaching = next(e for e in metadata["edges"] if e["family"] == "negative_error")
    forward = SimpleNamespace(event=(negative, action["terminal"], .25))
    feedback = SimpleNamespace(event=(negative, teaching["terminal"], .25))
    branch.network.presynaptic_wheel[0] = [forward, feedback]
    gate.before_step()
    assert branch.network.presynaptic_wheel[0] == [feedback]
    assert gate.removed == gate.observed == 1
    check_buffer_aliases(branch.network)


def test_native_signed_return_can_leave_nonnegative_credit_domain():
    cell = TerminalCreditNeuron(1, NeuronParameters(num_inputs=2, eta_post=1e-8, eta_retro=1e-6), log_level="CRITICAL")
    for sid in (0, 1):
        composition.synapse(cell, sid, 1.)
    cell.add_axon_terminal(100, 1)
    cell.register_source(1, 2, 100)
    cell.configure_terminal_credit([dict(name="alpha1", terminals=[100], receptor_ports=[1], receptor_weights=[1.])])
    cell.presynaptic_points[100].u_o.info = 5e-7
    cell.process_retrograde_signal(RetrogradeSignalEvent(2, 0, 1, 100, np.array([-1., 0., 0., 0.]), 0))
    assert cell.presynaptic_points[100].u_o.info < 0
    # Characterize the existing composition conflict; do not silently clip it.
    with pytest.raises(ValueError, match="Invalid eligibility flow"):
        cell.tick({}, 0)
