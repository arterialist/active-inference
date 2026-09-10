"""Check the signed composition's domain and unchanged positive behavior."""
import copy

import numpy as np

from simulations.drosophila.memory_feedback import ingestion_comparison as wiring
from simulations.drosophila.memory_feedback.terminal_credit import TerminalCreditNeuron
from simulations.drosophila.memory_feedback.signed_terminal_credit import SignedTerminalCreditNeuron
from neuron.neuron import Neuron, NeuronParameters, RetrogradeSignalEvent


def cell(cls):
    neuron = cls(1, NeuronParameters(num_inputs=2, eta_post=1e-8, eta_retro=1e-6, c=3), log_level="CRITICAL")
    for sid in (0, 1):
        wiring.synapse(neuron, sid, 1.)
    neuron.add_axon_terminal(100, 1)
    neuron.presynaptic_points[100].u_o.mod[:] = 0.
    neuron.register_source(1, 2, 100)
    neuron.configure_terminal_credit([dict(name="alpha1", terminals=[100], receptor_ports=[1], receptor_weights=[1.])])
    return neuron


def test_positive_course_preserves_original_arithmetic_and_state():
    original = cell(TerminalCreditNeuron)
    signed = copy.deepcopy(original)
    signed.__class__ = SignedTerminalCreditNeuron
    for tick in range(100):
        for neuron in (original, signed):
            neuron.input_buffer[0, 0] = 40. if tick < 50 else 0.
            neuron.input_buffer[1, 0] = 1. if 20 <= tick < 70 else 0.
            neuron.tick({}, tick)
        for name in ("S", "O", "r", "b", "t_ref", "F_avg", "terminal_credit_kc", "terminal_credit_updates"):
            assert getattr(original, name) == getattr(signed, name)
        assert original.presynaptic_points[100].u_o.info == signed.presynaptic_points[100].u_o.info
        assert original.terminal_credit_groups[0]["dopamine"] == signed.terminal_credit_groups[0]["dopamine"]


def test_signed_return_crossing_is_preserved_and_credit_contracts_magnitude():
    neuron = cell(SignedTerminalCreditNeuron)
    terminal = neuron.presynaptic_points[100]
    terminal.u_o.info = 5e-7
    neuron.process_retrograde_signal(RetrogradeSignalEvent(2, 0, 1, 100, np.array([-1., 0., 0., 0.]), 0))
    negative = terminal.u_o.info
    assert negative == -5e-7
    neuron.terminal_credit_groups[0]["dopamine"] = 1.
    neuron.terminal_credit_kc = .5
    neuron.S = 2.
    neuron.input_buffer[0, 0] = 40.
    events = neuron.tick({}, 0)
    assert negative < terminal.u_o.info < 0
    assert next(e[2] for e in events if isinstance(e, tuple) and e[1] == 100) == negative
    neuron.process_retrograde_signal(RetrogradeSignalEvent(2, 0, 1, 100, np.array([1., 0., 0., 0.]), 1))
    assert terminal.u_o.info > 0  # Native return can recover the positive domain.


def test_native_negative_information_is_not_inhibitory_release():
    receiver = Neuron(2, NeuronParameters(num_inputs=2, eta_post=1e-8, eta_retro=1e-6), log_level="CRITICAL")
    wiring.synapse(receiver, 0, 1.)
    wiring.synapse(receiver, 1, 0.)
    receiver.input_buffer[0, 0] = -.5
    receiver.tick({}, 0)
    assert receiver.S == 0 and not receiver.propagation_queue
