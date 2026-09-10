"""One source-specific efficacy test following the measured recruitment failure.

Dry A intact-minus-cut somatic records predict first student-DAN recruitment
at gain 60.13565645842552 under fixed other inputs. Test the next power of two,
64, from birth. This scales only the existing SMP108-to-PAM07/PAM08 receiving
information coefficients. Counts, sign, identities, reciprocal connections,
all other efficacies, native thresholds and adaptation remain as specified.
It is an explicit efficacy hypothesis, not an anatomical count correction or
a physiological calibration. The coupled simulation tests the linear estimate.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import feeding, student_course, terminal_course
from ..connectome import sha256


def build(graph):
    prep, selected = terminal_course.build(graph)
    roles = student_course.groups(graph)
    sources = {prep.root_to_id[r] for r in roles["SMP108"]}
    targets = {prep.root_to_id[r] for role in ("PAM07", "PAM08") for r in roles[role]}
    changes = []
    for row, pre, terminal, post, sid in prep.edge_bindings:
        if int(pre) in sources and int(post) in targets:
            cell = prep.network.network.neurons[int(post)]
            point = cell.postsynaptic_points[int(sid)]
            before = float(point.u_i.info); after = 64.*before
            if before <= 0 or after > cell.params.w_max:
                raise ValueError("Efficacy hypothesis exceeds the native receiving bound")
            point.u_i.info = after
            changes.append(dict(source_row=int(row), source_id=int(pre), target_id=int(post),
                                synapse_id=int(sid), original_weight=before, effective_weight=after))
    prep.assumptions["feedback_efficacy"] = dict(gain=64., changed_pairs=changes,
        derivation="Next power of two above held-input first-crossing estimate 60.13565645842552, from native dry learned-A projection records before this simulation",
        calibration_records="memory-terminal-projection-paired-20260910/A-intact.npz and A-cut.npz",
        calibrated_target_root="720575940605280201", calibration_probe_tick=198,
        interpretation="Unfitted source-specific current efficacy, not synapse count, receptor identification or known fly physiological gain",
        source_sha256=sha256(Path(__file__)))
    return prep, selected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true")
    a=p.parse_args(); original=feeding.configure
    def configured(graph, *, sensitive=True):
        with patch.object(feeding, "configure", original), patch.object(feeding, "memory_build", build):
            return feeding.configure(graph)
    with patch.object(feeding, "configure", configured), patch.object(feeding, "groups", student_course.groups):
        feeding.run(a.graph,a.output,paired=not a.unpaired)
