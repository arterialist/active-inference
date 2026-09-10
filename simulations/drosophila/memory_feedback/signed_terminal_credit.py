"""Separate signed-information version of the terminal-credit composition.

Native presynaptic return keeps q in [-100, 100], and may cross zero. For
nonnegative receptor drive a=eta*d*x, the pure-depression flow dq/dt=-a*q is
well-defined throughout that interval. It contracts |q| and preserves sign;
native return still supplies its unmodified additive update and signed bounds.
On positive q we reuse the original exact-flow arithmetic. On negative q we
apply its odd extension. No excursion is clipped, discarded or rate-limited.

q is now explicitly a signed information coefficient, not nonnegative release
probability or transmitter quantity. Native emission retains signed q, but the
base receiver processes only positive information arrivals. Thus a negative
terminal is ineffective on a single-source ordinary receiving port; it is NOT
biological inhibitory release. This version preserves that native behavior.
The originally measured connection signs do not establish a biological model
of such negative terminal states. Dopamine receptors still require actual
nonnegative source arrivals; their interpretation is not changed here.

The accepted TerminalCreditNeuron and its saved runs remain unchanged. Local
receptor traces, native return, receiving adaptation, and pre-update emission
order are identical. This is a changed composition domain, not a new default.
"""
import math

import numpy as np

from .terminal_credit import TerminalCreditNeuron
from .ingestion_comparison import serialized
from neuron.neuron import Neuron, MAX_SYNAPTIC_WEIGHT
from neuron.extensions.experimental.eligibility_trace import eligibility_step


def signed_depression_step(q, minus, eta, cap):
    q = np.asarray(q)
    magnitude = eligibility_step(np.abs(q), 0., minus, eta, cap)
    return np.copysign(magnitude, q)


class SignedTerminalCreditNeuron(TerminalCreditNeuron):
    __slots__ = ()

    def tick(self, external_inputs, current_tick, dt=1.):
        groups = getattr(self, "terminal_credit_groups", ())
        if not groups or not self.terminal_credit_enabled:
            return Neuron.tick(self, external_inputs, current_tick, dt)
        if dt != 1 or (self.terminal_credit_tick is not None and current_tick != self.terminal_credit_tick+1):
            raise ValueError("Terminal traces require consecutive unit ticks")
        pending = []
        for group in groups:
            if any(int(p) in external_inputs for p in group["ports"]):
                raise ValueError("Dopamine receptors must be driven by measured neural sources")
            incoming = self.input_buffer[group["ports"], 0]
            if not np.isfinite(incoming).all() or np.any(incoming < 0):
                raise ValueError("Invalid dopamine-source release amplitude")
            before = np.array([self.presynaptic_points[t].u_o.info for t in group["terminals"]])
            after = signed_depression_step(before, group["dopamine"]*self.terminal_credit_kc,
                                           self.terminal_credit_eta, MAX_SYNAPTIC_WEIGHT)
            pending.append((group, before, after, float(np.dot(incoming, group["weights"]))))
        events = Neuron.tick(self, external_inputs, current_tick, dt)
        d_da = math.exp(-1/self.terminal_credit_tau_da)
        for group, before, after, arrival in pending:
            for terminal, value in zip(group["terminals"], after, strict=True):
                self.presynaptic_points[terminal].u_o.info = float(value)
            self.terminal_credit_updates += int(np.count_nonzero(before != after))
            group["dopamine"] = d_da*group["dopamine"]+(1-d_da)*arrival
        d_kc = math.exp(-1/self.terminal_credit_tau_kc) if self.terminal_credit_tau_kc else 0.
        self.terminal_credit_kc = d_kc*self.terminal_credit_kc+(1-d_kc)*float(self.O > 0)
        self.terminal_credit_tick = current_tick
        return events


def convert(branch):
    """Change only the class implementing the declared flow, preserving state."""
    cells = [n for n in branch.network.network.neurons.values() if type(n) is TerminalCreditNeuron]
    before = serialized(branch)
    for cell in cells:
        cell.__class__ = SignedTerminalCreditNeuron
    for cell in cells:
        cell.__class__ = TerminalCreditNeuron
    exact = serialized(branch) == before
    assert exact
    for cell in cells:
        cell.__class__ = SignedTerminalCreditNeuron
    return dict(converted_cells=len(cells), all_other_state_and_rng_exact=exact,
        domain="Signed information coefficient; native return and positive-arrival receiving filter retained. Credit contracts magnitude on either sign. Negative q is not inhibitory transmitter release.")
