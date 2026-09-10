"""Delayed local credit must preserve native release order and other terminals."""
import copy

import numpy as np

from simulations.drosophila.memory_feedback.terminal_credit import TerminalCreditNeuron
from neuron.neuron import Neuron, NeuronParameters, PostsynapticPoint, PostsynapticInputVector, PresynapticPoint, PresynapticOutputVector, RetrogradeSignalEvent


def cell(cls=TerminalCreditNeuron, enabled=True, tau=64.):
    n = cls(1, NeuronParameters(num_inputs=3, eta_post=1e-8, eta_retro=1e-6, c=3), log_level="CRITICAL")
    for sid, weight in enumerate((1., .02, .02)):
        n.postsynaptic_points[sid] = PostsynapticPoint(PostsynapticInputVector(info=weight, plast=0., adapt=np.zeros(2)))
        n.distances[sid] = 0
    for terminal in (0, 1):
        n.presynaptic_points[terminal] = PresynapticPoint(PresynapticOutputVector(info=1., mod=np.zeros(2)), 1.)
    n.register_source(1, 2, 0); n.register_source(2, 3, 0)
    if cls is TerminalCreditNeuron:
        n.configure_terminal_credit([
            dict(name="alpha1", terminals=[0], receptor_ports=[1], receptor_weights=[2.]),
            dict(name="gamma4", terminals=[1], receptor_ports=[2], receptor_weights=[1.])], enabled=enabled, tau_kc=tau)
    return n


def course(n, cue=(0, 40), dopamine=(60, 80)):
    releases = []
    for t in range(160):
        n.input_buffer[0, 0] = 40. if cue[0] <= t < cue[1] else 0.
        n.input_buffer[1, 0] = 1. if dopamine[0] <= t < dopamine[1] else 0.
        before = n.presynaptic_points[0].u_o.info
        events = n.tick({}, t)
        for event in events:
            if isinstance(event, tuple) and event[1] == 0:
                assert event[2] == before  # Native emitted events precede new credit.
                releases.append(event)
    return n.presynaptic_points[0].u_o.info, releases


def test_delayed_credit_is_local_and_order_sensitive():
    forward = cell(); q, spikes = course(forward)
    instantaneous = cell(tau=0.); qi, _ = course(instantaneous)
    reverse = cell(); qr, _ = course(reverse, cue=(60, 100), dopamine=(0, 20))
    absent = cell(); qa, _ = course(absent, dopamine=(0, 0))
    assert spikes and 0 < q < .9
    assert qi == qa == 1.
    assert qr > .999
    assert forward.presynaptic_points[1].u_o.info == 1.
    assert forward.postsynaptic_points[1].u_i.info != .02  # Native receiving adaptation.


def test_disabled_extension_matches_base_and_returning_learning_stays_active():
    a = cell(); a.terminal_credit_enabled = False
    b = cell(Neuron)
    assert course(a)[0] == course(b)[0]
    for field in ("S", "O", "r", "b", "t_ref", "F_avg"):
        assert getattr(a, field) == getattr(b, field)
    for sid in a.postsynaptic_points:
        assert a.postsynaptic_points[sid].u_i.info == b.postsynaptic_points[sid].u_i.info
    active = cell()
    active.process_retrograde_signal(RetrogradeSignalEvent(4, 0, 1, 0, np.array([.5, 0., 0., 0.]), 0))
    assert active.presynaptic_points[0].u_o.info > 1.


def test_trace_state_continues_across_copy_without_reset():
    a = cell()
    for t in range(30):
        a.input_buffer[0, 0] = 40.
        a.tick({}, t)
    b = copy.deepcopy(a)
    for t in range(30, 100):
        for n in (a, b):
            n.input_buffer[1, 0] = float(60 <= t < 80)
            n.tick({}, t)
        assert a.presynaptic_points[0].u_o.info == b.presynaptic_points[0].u_o.info
        assert a.terminal_credit_kc == b.terminal_credit_kc
