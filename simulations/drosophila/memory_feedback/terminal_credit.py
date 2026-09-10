"""Opt-in dopamine-paired eligibility at identified KC output terminals.

The failed selected-input rule cannot assign delayed credit without another
input arrival. Its dopamine-independent negative term also erased weak student
inputs. Existing spike-pair eligibility requires an MBON spike; the existing
graded predictor requires different receptor wiring and release dynamics.

This separate composition reuses eligibility_step's exact bounded flow:
    q' = q * exp(-eta_credit * previous_dopamine * previous_KC_trace).
The two traces use the existing exponential receptor/context update. KC trace
is an EMA of actual KC spikes; dopamine trace is an EMA of actual arrivals at
declared measured DAN-to-KC receptors, weighted by their birth contact counts.
Each receptor group affects only declared KC-to-MBON terminals in its matching
compartment. Native receiving learning and native returning adaptation continue.
Native output events use pre-update coefficients. No cue name, body reward,
training flag, target response or external dopamine state enters this class.

This is a NEW phenomenological composition, not a new default or a molecular
model. Hige et al. 2015, doi:10.1016/j.neuron.2015.11.003, motivates KC-output
depression independent of MBON spiking in gamma1pedc, not these gamma4/alpha1
kinetics. Normalizing receptor contact weights to sum one is an explicit
unfitted potency assumption. The current two-point graph does not locate
individual receptors within KC axonal compartments.
"""
import math

import numpy as np

from simulations.paula_loader import ensure_paula_available
ensure_paula_available()
from neuron.neuron import Neuron, MAX_SYNAPTIC_WEIGHT
from neuron.extensions.experimental.eligibility_trace import eligibility_step


class TerminalCreditNeuron(Neuron):
    def configure_terminal_credit(self, groups, *, enabled=True, tau_kc=64., tau_da=4., eta=.1):
        if hasattr(self, "terminal_credit_groups"):
            raise RuntimeError("Terminal credit is configured once, at birth")
        if not all(math.isfinite(v) for v in (tau_kc, tau_da, eta)) or tau_kc < 0 or tau_da <= 0 or eta <= 0:
            raise ValueError("Need nonnegative KC memory and positive receptor decay/rate")
        if self.params.eta_post <= 0 or self.params.eta_retro <= 0 or self._ablation:
            raise ValueError("Native adaptation must remain enabled")
        assigned = set(); configured = []
        for group in groups:
            terminals = tuple(group["terminals"])
            ports = np.asarray(group["receptor_ports"], dtype=int)
            strengths = np.asarray(group["receptor_weights"], dtype=float)
            if (not terminals or not len(ports) or len(set(terminals)) != len(terminals)
                    or len(set(ports.tolist())) != len(ports) or assigned.intersection(terminals)
                    or any(t not in self.presynaptic_points for t in terminals)
                    or any(int(p) not in self.synapse_sources for p in ports)
                    or strengths.shape != ports.shape or not np.isfinite(strengths).all()
                    or np.any(strengths <= 0)):
                raise ValueError("Need distinct measured neural receptors and matching terminals")
            assigned.update(terminals)
            configured.append(dict(name=group["name"], terminals=terminals, ports=ports,
                                   weights=strengths/strengths.sum(), dopamine=0.))
        self.terminal_credit_groups = configured
        self.terminal_credit_enabled = bool(enabled)
        self.terminal_credit_kc = 0.
        self.terminal_credit_tau_kc = tau_kc
        self.terminal_credit_tau_da = tau_da
        self.terminal_credit_eta = eta
        self.terminal_credit_tick = None
        self.terminal_credit_updates = 0

    def tick(self, external_inputs, current_tick, dt=1.):
        groups = getattr(self, "terminal_credit_groups", ())
        if not groups or not self.terminal_credit_enabled:
            return super().tick(external_inputs, current_tick, dt)
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
            after = eligibility_step(before, 0., group["dopamine"]*self.terminal_credit_kc,
                                     self.terminal_credit_eta, MAX_SYNAPTIC_WEIGHT)
            pending.append((group, before, after, float(np.dot(incoming, group["weights"]))))
        events = super().tick(external_inputs, current_tick, dt)
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
