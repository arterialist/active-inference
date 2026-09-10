"""Embodied direct-learning check for measured DAN-to-KC terminal credit.

The reference and paired/unpaired courses use the same expanded measured graph,
the same native weak positive learning rates and output/physical boundaries.
Only the explicit terminal-credit mechanism and primary nutrient timing vary.
No reward_hebb receiving rule or artificial student-DAN sensitivity is used.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import feeding, student_course
from .terminal_credit import TerminalCreditNeuron
from .. import paula
from ..connectome import sha256


def build(graph, *, enabled=True, tau_kc=64.):
    with patch.object(paula, "Neuron", TerminalCreditNeuron):
        prep = paula.build_paula(graph, paula.Dynamics())
    annotation = {prep.root_to_id[r]: graph.nodes[r]["annotation"] for r in graph.selected}
    counts = {int(e[8]): int(e[4]) for e in graph.internal}
    selected, configurations, uncovered = [], [], []
    for root in graph.selected:
        nid = prep.root_to_id[root]; a = annotation[nid]
        if a["cell_class"] != "Kenyon_Cell":
            continue
        groups = []
        for compartment, target, providers in (("alpha1", "MBON07", {"PAM11"}), ("gamma4", "MBON04", {"PAM07", "PAM08"})):
            outputs = [e for e in prep.edge_bindings if int(e[1]) == nid and annotation[int(e[3])]["hemibrain_type"] == target]
            if not outputs:
                continue
            selected.extend(outputs)
            inputs = [e for e in prep.edge_bindings if int(e[3]) == nid and annotation[int(e[1])]["hemibrain_type"] in providers]
            if not inputs:
                uncovered.append(dict(root=root, compartment=compartment, terminals=[int(e[2]) for e in outputs]))
                continue
            group = dict(name=compartment, terminals=[int(e[2]) for e in outputs],
                receptor_ports=[int(e[4]) for e in inputs], receptor_weights=[counts[int(e[0])] for e in inputs])
            groups.append(group)
            configurations.append(dict(root=root, **group, receptor_source_rows=[int(e[0]) for e in inputs],
                                       output_source_rows=[int(e[0]) for e in outputs]))
        if groups:
            prep.network.network.neurons[nid].configure_terminal_credit(groups, enabled=enabled, tau_kc=tau_kc)
    prep.assumptions["terminal_credit"] = dict(enabled=enabled, tau_kc=tau_kc, tau_dopamine=4., eta_credit=.1,
        groups=configurations, uncovered_outputs=uncovered,
        native="Unchanged weak positive receiving and returning adaptation. No selected reward_hebb rule. Emissions use pre-update release.",
        dopamine="Actual PAM11-to-KC or PAM07/PAM08-to-KC arrivals, contact-weighted and normalized per local receptor group. No DAN-to-MBON modulation surrogate.",
        limit="Phenomenological terminal-group eligibility and receptor kinetics. Pair graph does not locate contacts within KC axonal compartments; no efficacy or timing calibration to flies.",
        implementation_sha256=sha256(Path(__file__).with_name("terminal_credit.py")),
        configuration_sha256=sha256(Path(__file__)))
    return prep, np.asarray(selected, dtype=np.int64).reshape(-1, 5)


def configure(graph, *, enabled=True, tau_kc=64.):
    with patch.object(feeding, "memory_build", lambda g: build(g, enabled=enabled, tau_kc=tau_kc)):
        return feeding.configure(graph)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--disabled", action="store_true"); p.add_argument("--unpaired", action="store_true")
    p.add_argument("--instantaneous", action="store_true", help="KC spike eligibility lasts one tick only")
    a = p.parse_args(); original = feeding.configure
    def configured(graph, *, sensitive=True):
        with patch.object(feeding, "configure", original):
            return configure(graph, enabled=not a.disabled, tau_kc=0. if a.instantaneous else 64.)
    with patch.object(feeding, "configure", configured), patch.object(feeding, "groups", student_course.groups):
        feeding.run(a.graph, a.output, paired=not a.unpaired)
