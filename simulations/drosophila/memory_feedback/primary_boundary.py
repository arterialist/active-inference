"""Test broader primary PAM recruitment after the student-output conflict.

An alpha1-only primary boundary leaves A's gamma4 output untrained. Once that
output spikes, its measured inhibition suppresses SMP108 and prevents A from
recruiting student dopamine. Test the same nutrient-to-current transducer on
the represented PAM11, PAM07 and PAM08 groups. It operates whenever nutrients
are ingested, regardless of cue or phase, and is zero otherwise.

PAM-cluster stimulation during first-order pairing in Yamada et al. 2023,
Figure 5, motivates the comparison (https://pmc.ncbi.nlm.nih.gov/articles/PMC9937650/).
The chosen type coverage and equal current are explicit engineering boundaries,
not a reconstruction of that genetic driver or a calibrated sugar pathway.
All second-order acquisition remains nutrient-free and neurally mediated.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import feeding, student_course, student_output, continuing_course, second_order
from ..connectome import sha256

NUTRIENT_ROLES = ("PAM11", "PAM07", "PAM08")


def step(net, ids, roles, codes, cue, organism, *, factor, pump=False, well=False, advance=None):
    pump_j, well_j = organism.offer(pump, well)
    active = set(codes.get(cue, ()))
    for code in codes.values():
        for root in code:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 40. if root in active else 0.)
    for role in NUTRIENT_ROLES:
        for root in roles[role]:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 40.*(pump_j+well_j)/feeding.DOSE_J)
    for root in roles["SMP353"]:
        cell = net.network.neurons[ids[root]]
        net.set_external_input(cell.id, cell.params.num_inputs-1, 1.4/factor if active else 0.)
    (advance or net.run_tick)()
    spike = float(np.mean([net.network.neurons[ids[r]].O for r in roles["SMP353"]]))
    return organism.step(spike, pump_j, well_j)


def build(graph):
    prep, selected = student_output.build(graph)
    roles = student_course.groups(graph)
    prep.assumptions["primary_boundary"] = dict(roles=list(NUTRIENT_ROLES),
        roots=[r for role in NUTRIENT_ROLES for r in roles[role]], current_per_dose=40.,
        mechanism="Same actual ingested-energy transducer, broader represented PAM targets; no cue or phase dependence",
        source_sha256=sha256(Path(__file__)),
        motivation="Causal MBON04-to-SMP108 block restores the A teacher. First-order PAM-cluster stimulation in Yamada et al. Figure 5 motivates testing broader recruitment.",
        reference="https://pmc.ncbi.nlm.nih.gov/articles/PMC9937650/",
        limit="Equal current and type coverage are engineered, not measured nutrient sensory wiring or reconstructed driver expression")
    return prep, selected


def run_course(graph, output, *, unpaired=False):
    original = feeding.configure
    def configured(graph, *, sensitive=True):
        with patch.object(feeding, "configure", original), patch.object(feeding, "memory_build", build):
            return feeding.configure(graph)
    with patch.object(feeding, "configure", configured), patch.object(feeding, "groups", student_course.groups), patch.object(feeding, "step", step):
        return feeding.run(graph, output, paired=not unpaired)


def run_continuation(receiver, output, *, cut=False, displaced=False):
    # Keep the same physical nutrient boundary during diagnostic A feeding.
    # Acquisition offers no nutrients, hence every primary DAN input is zero.
    manifest = json.loads((Path(receiver)/"manifest.json").read_text())
    boundary = manifest["assumptions"]["primary_boundary"]
    if boundary["roles"] != list(NUTRIENT_ROLES) or boundary["source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Primary boundary differs from the recorded parent")
    with patch.object(second_order, "step", step):
        result = continuing_course.run(receiver, output, cut=cut, displaced=displaced)
    result["primary_boundary_sha256"] = sha256(Path(__file__))
    (Path(output)/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true"); p.add_argument("--continue-course", action="store_true")
    p.add_argument("--cut", action="store_true"); p.add_argument("--displaced", action="store_true")
    a=p.parse_args()
    if a.continue_course:
        if a.unpaired: p.error("Choose the unpaired parent directory for continuation")
        run_continuation(a.input, a.output, cut=a.cut, displaced=a.displaced)
    else:
        if a.cut or a.displaced: p.error("Projection/timing interventions apply only during continuation")
        run_course(a.input, a.output, unpaired=a.unpaired)
