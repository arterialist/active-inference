"""Opt-in input selection for two EXISTING PAULA learning equations.

Motivation: whole-MBON reward_hebb in the first recorded comparison reversed
the imported signs of reciprocal MBON and APL currents. NeuronParameters has one
rule per cell; PortModulationNeuron only adjusts native-rule rate. Here selected
KC->MBON ports use exactly the existing reward_hebb update, while all other ports
retain the native multiplicative rule and its positive learning rate. No new
weight equation, eligibility state, behavioral target or dopamine trace is added.

As in existing PAULA experimental extensions, capture selected pre-update input,
execute the inherited tick, then replace only selected receiving info weights.
Propagation and returning errors retain their pre-update values and native order.
Empty memory_rule_ports follows the unmodified inherited tick. This is a test of
where a rule applies, not an identified molecular or terminal-compartment model.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import acquisition
from .. import paula
from neuron.neuron import Neuron


class MemoryInputRuleNeuron(Neuron):
    def tick(self, external_inputs, current_tick, dt=1.):
        ports = self.metadata.get("memory_rule_ports", ())
        if not ports:
            return super().tick(external_inputs, current_tick, dt)
        if self.params.plasticity_mode != "legacy_multiplicative" or self._ablation:
            raise ValueError("Selected-rule experiment requires unablated native background")
        if not 0 < self.params.eta_post <= 1 or self.params.eta_retro <= 0:
            raise ValueError("Require positive basal adaptation and eta_post <= 1")
        pending = [(sid, float(self.postsynaptic_points[sid].u_i.info),
                    float(self.input_buffer[sid, 0])) for sid in ports
                   if self.input_buffer[sid, 0] > 0]
        if any(w < 0 or self.postsynaptic_points[sid].u_i.plast != 0 for sid, w, _ in pending):
            raise ValueError("Selected memory inputs must be excitatory with zero plastic throughput")
        events = super().tick(external_inputs, current_tick, dt)
        direction = 1. if current_tick-self.t_last_fire <= self.t_ref else -1.
        mr = self.M_vector[self.params.nm_reward_index]
        ms = self.M_vector[self.params.nm_stress_index]
        nm = max(0., 1.+self.params.nm_plasticity_kappa*(mr-ms))
        for sid, before, info in pending:
            value = before + self.params.eta_post*(nm*direction*info-self.params.rh_decay*before)
            self.postsynaptic_points[sid].u_i.info = float(np.clip(value, self.params.w_min, self.params.w_max))
        return events


def build(graph, rule="dopamine_hebb", dopamine=True):
    # Patch only the constructor name used by the existing graph builder, for
    # this local construction. Other preparations and shared files are intact.
    with patch.object(paula, "Neuron", MemoryInputRuleNeuron):
        prep, selected = acquisition.build(graph, "native", dopamine)
    for root in acquisition.groups(graph)["MBON07"]:
        cell = prep.network.network.neurons[prep.root_to_id[root]]
        ports = selected[selected[:, 3] == cell.id, 4].tolist()
        cell.metadata["memory_rule_ports"] = ports
        cell.params.nm_plasticity_kappa = -10.
        cell.params.rh_decay = 1.
    prep.assumptions["learning_comparison"].update(
        rule="input_selected_existing_reward_hebb", memory_rule_ports=selected[:, [3, 4]].tolist(),
        equation="Unchanged reward_hebb at selected KC->MBON inputs, legacy_multiplicative everywhere else",
        extension_source=str(Path(__file__).resolve()),
        reason="Whole-cell alternative reversed reciprocal current signs in the first comparison",
        limit="An explicit rule-location hypothesis; still soma-wide dopamine and no biological timing fit")
    return prep, selected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true")
    p.add_argument("--no-dopamine-receptor", action="store_true")
    a = p.parse_args()
    original = acquisition.build
    def selected_builder(graph, rule, dopamine):
        # Avoid recursion when acquisition.run calls its temporarily replaced
        # factory. Restore its original name during this synchronous build.
        with patch.object(acquisition, "build", original):
            return build(graph, rule, dopamine)
    with patch.object(acquisition, "build", selected_builder):
        acquisition.run(a.graph, a.output, rule="dopamine_hebb", paired=not a.unpaired,
                        dopamine=not a.no_dopamine_receptor, blocks=2)
