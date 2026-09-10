"""Test a uniform feedback efficacy of 256 from birth to broaden coverage.

The prior gain of 64 recruited two measured providers during learned A and
changed only five B terminals after eight pairings. A fixed fourfold projection
of that terminal depression still did not change B action. Held-input estimates
predict that the next coverage bound, at least half of every cue's anatomical
KC-to-MBON04 contact mass, needs four times the existing source efficacy.

This is an unfitted engineering hypothesis. It changes only existing
SMP108-to-PAM07/PAM08 receiving coefficients, uniformly and from birth.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import student_interface
from ..connectome import sha256

_original_build=student_interface.build


def scale(prep, roles):
    sources={prep.root_to_id[r] for r in roles["SMP108"]}
    targets={prep.root_to_id[r] for role in ("PAM07","PAM08") for r in roles[role]}
    pending=[]
    for row,pre,terminal,post,sid in prep.edge_bindings:
        if int(pre) not in sources or int(post) not in targets: continue
        cell=prep.network.network.neurons[int(post)];point=cell.postsynaptic_points[int(sid)]
        before=float(point.u_i.info);after=4*before
        if not 0<after<=cell.params.w_max: raise ValueError("Coverage efficacy exceeds native coefficient bound")
        pending.append((point,dict(source_row=int(row),source_id=int(pre),target_id=int(post),
                                  synapse_id=int(sid),before=before,after=after)))
    if not pending: raise ValueError("No measured feedback pairs")
    for point,r in pending: point.u_i.info=r["after"]
    return [r for point,r in pending]


def build(graph):
    from .student_course import groups
    prep,selected=_original_build(graph)
    changes=scale(prep,groups(graph))
    prep.assumptions["coverage_efficacy"]=dict(multiplier_on_prior=4,total_birth_gain=256,
        changed_pairs=changes,source_sha256=sha256(Path(__file__)),
        derivation="recruitment-coverage.json: smallest doubling covering at least half of each cue's KC-to-MBON04 contact mass under held-input recruitment estimates",
        analysis_sha256=sha256(Path(__file__).with_name("evidence")/"recruitment-coverage.json"),
        limit="Uniform unfitted efficacy hypothesis at measured existing pairs. Counts, signs, graph, native adaptation, credit kinetics, sensory and motor interfaces remain fixed. No biological gain calibration.")
    return prep,selected


def run(graph,output,*,unpaired=False):
    with patch.object(student_interface,"build",build):
        return student_interface.run_course(graph,output,unpaired=unpaired)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("graph",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--unpaired",action="store_true");a=p.parse_args();run(a.graph,a.output,unpaired=a.unpaired)
