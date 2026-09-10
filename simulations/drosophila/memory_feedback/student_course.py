"""Check that alpha1 memory and its teaching output survive gamma4 composition.

No reward is assigned to the student DANs. They can be recruited only through
retained neural inputs. Their measured connections to MBON04 carry the same
explicit dopamine-channel hypothesis as the original PAM11-to-MBON07 pairs.
Run the existing complete acquisition/feeding protocol before extending to B.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import feeding
from . import input_rule
from .nonnegative_rule import NonnegativeMemoryInputNeuron
from .acquisition import groups as parent_groups
from ..connectome import sha256


def groups(graph):
    out = parent_groups(graph)
    for name in ("MBON04", "PAM07", "PAM08"):
        out[name] = [r for r in graph.selected if graph.nodes[r]["annotation"]["hemibrain_type"] == name]
    return out


def configure(graph, *, sensitive=True):
    with patch.object(input_rule, "MemoryInputRuleNeuron", NonnegativeMemoryInputNeuron):
        prep, teacher, factor = feeding.configure(graph, sensitive=sensitive)
    roles = groups(graph)
    mbons = {prep.root_to_id[r] for r in roles["MBON04"]}
    dans = {prep.root_to_id[r] for r in roles["PAM07"]+roles["PAM08"]}
    kcs = {prep.root_to_id[r] for r in graph.selected if graph.nodes[r]["annotation"]["cell_class"] == "Kenyon_Cell"}
    selected = []; modulation = []
    for row, pre, terminal, post, sid in prep.edge_bindings:
        if pre in dans and post in mbons:
            cell = prep.network.network.neurons[int(pre)]
            cell.presynaptic_points[int(terminal)].u_o.mod[1] = 1.
            p = prep.network.network.neurons[int(post)].postsynaptic_points[int(sid)]
            p.u_i.adapt[1] = abs(p.u_i.info)
            modulation.append(int(row))
        if pre in kcs and post in mbons:
            selected.append((int(row),int(pre),int(terminal),int(post),int(sid)))
    selected = np.array(selected,dtype=np.int64).reshape(-1,5)
    for nid in mbons:
        cell = prep.network.network.neurons[nid]
        cell.metadata["memory_rule_ports"] = selected[selected[:,3]==nid,4].tolist()
        cell.params.eta_post = .01; cell.params.nm_plasticity_kappa = -10.; cell.params.rh_decay = 1.
    prep.assumptions["student_compartment"] = dict(
        MBON04_roots=roles["MBON04"],DAN_roots=roles["PAM07"]+roles["PAM08"],
        dopamine_source_rows=modulation, memory_pairs=len(selected),
        input="Controlled A/B/C gamma-KC codes, with all measured internal connections",
        reward="No external stimulation or modulation of student DANs; only neural afferents",
        memory_bound="Existing zero lower bound applied only to selected excitatory memory inputs; inhibitory reciprocal inputs retain native signed adaptation",
        source_sha256=sha256(Path(__file__)),
        limitation="Native student-DAN input integration and thresholds; instantaneous selected-port plasticity. This composition is tested, not assumed to work.")
    return prep, np.concatenate((teacher,selected)), factor


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--unpaired",action="store_true")
    a=p.parse_args();original=feeding.configure
    def configured(graph,*,sensitive=True):
        with patch.object(feeding,"configure",original):
            return configure(graph,sensitive=sensitive)
    with patch.object(feeding,"configure",configured),patch.object(feeding,"groups",groups):
        feeding.run(a.graph,a.output,paired=not a.unpaired)
