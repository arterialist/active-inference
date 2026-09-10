"""Test explicit cue-input and physical-output boundaries at SMP108.

The retained balanced preparation cannot express even imposed complete gamma4
B depression: SMP108 receives no experimental cue-presence drive and the body
reads SMP353. Apply the existing equal cue-presence current to SMP108 as well,
and use its activity for the same muscle/hinge transducer. These are engineered
interfaces, not added anatomical synapses or reconstructed fly motor wiring.

Yamada et al. 2023 reports odor responses and lateral-horn inputs at SMP108
(Figure 5), and activation-evoked upwind steering (Figure 7 supplement 1):
https://elifesciences.org/articles/79042/figures
SMP108 was dispensable for the tested conditioned steering, so its use as the
physical readout here is an assay choice, not a necessary fly motor pathway.
"""
import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import feeding, primary_boundary, student_expression
from ..connectome import sha256

_primary_build = primary_boundary.build


def step(net, ids, roles, codes, cue, organism, *, factor, pump=False, well=False, advance=None, motor_role="SMP108"):
    pump_j, well_j = organism.offer(pump, well)
    active = set(codes.get(cue, ()))
    for code in codes.values():
        for root in code:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 40. if root in active else 0.)
    for role in primary_boundary.NUTRIENT_ROLES:
        for root in roles[role]:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 40.*(pump_j+well_j)/feeding.DOSE_J)
    for role in ("SMP353", "SMP108"):
        for root in roles[role]:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 1.4/factor if active else 0.)
    (advance or net.run_tick)()
    spike = float(np.mean([net.network.neurons[ids[r]].O for r in roles[motor_role]]))
    return organism.step(spike, pump_j, well_j)


def build(graph):
    prep, selected = _primary_build(graph)
    prep.assumptions["student_interface"] = dict(cue_presence_targets=["SMP353", "SMP108"],
        cue_presence_current="1.4 / existing output sensitivity factor", motor_role="SMP108",
        source_sha256=sha256(Path(__file__)), reference="https://elifesciences.org/articles/79042/figures",
        limit="Equal presence input is an isolation boundary, not measured LH spikes. SMP108-to-hinge coupling is a physical assay, not reconstructed motor anatomy. All internal and incident boundary pairs remain unchanged.")
    return prep, selected


def run_course(graph, output, *, unpaired=False):
    with patch.object(primary_boundary, "build", build), patch.object(primary_boundary, "step", step):
        return primary_boundary.run_course(graph, output, unpaired=unpaired)


def run_continuation(receiver, output, *, cut=False, displaced=False):
    m = json.loads((Path(receiver)/"manifest.json").read_text())
    if m["assumptions"]["student_interface"]["source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Student interface changed since parent acquisition")
    with patch.object(primary_boundary, "step", step):
        result = primary_boundary.run_continuation(receiver, output, cut=cut, displaced=displaced)
    result["student_interface_sha256"] = sha256(Path(__file__))
    (Path(output)/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def probe(parent, output, *, zero=False, sensory_only=False):
    motor = "SMP353" if sensory_only else "SMP108"
    with patch.object(primary_boundary, "step", partial(step, motor_role=motor)):
        result = student_expression.run(parent, parent, output, zero=zero)
    result["interface_diagnostic"] = dict(added_cue_presence="SMP108", motor_role=motor,
        source_sha256=sha256(Path(__file__)),
        limit="Interfaces changed at expression on an existing parent. Imposed zero release is not learned memory; a full course from birth is required to assess the new preparation.")
    (Path(output)/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true"); p.add_argument("--continue-course", action="store_true")
    p.add_argument("--cut", action="store_true"); p.add_argument("--displaced", action="store_true")
    p.add_argument("--probe", action="store_true"); p.add_argument("--zero", action="store_true"); p.add_argument("--sensory-only", action="store_true")
    a=p.parse_args()
    if a.probe:
        if a.unpaired or a.continue_course or a.cut or a.displaced: p.error("Invalid diagnostic combination")
        probe(a.input, a.output, zero=a.zero, sensory_only=a.sensory_only)
    elif a.continue_course:
        if a.unpaired or a.zero or a.sensory_only: p.error("Invalid continuation combination")
        run_continuation(a.input, a.output, cut=a.cut, displaced=a.displaced)
    else:
        if a.cut or a.displaced or a.zero or a.sensory_only: p.error("Diagnostic flags require --probe or --continue-course")
        run_course(a.input, a.output, unpaired=a.unpaired)
