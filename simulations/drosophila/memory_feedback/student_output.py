"""Test whether native student output silence prevents memory expression.

Independent dry novel-B records at reference efficacy give a largest MBON04
potential of 0.5225614905357361 against threshold 1. The next power of two
above that first-crossing ratio, 1.9136503896886632, is two. Divide both native
MBON04 thresholds by two from birth, retaining their ratio. This is an
operating-point hypothesis, not measured fly excitability. All measured
connections and continuing learning remain, including inhibitory output to
SMP108. The source-specific feedback-efficacy hypothesis remains gain 64.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import feeding, feedback_efficacy, student_course
from ..connectome import sha256


def build(graph):
    prep, selected = feedback_efficacy.build(graph)
    changes = []
    for root in student_course.groups(graph)["MBON04"]:
        cell = prep.network.network.neurons[prep.root_to_id[root]]
        before = [cell.params.r_base, cell.params.b_base]
        cell.params.r_base /= 2.
        cell.params.b_base /= 2.
        cell.r = cell.params.r_base
        cell.b = cell.params.b_base
        changes.append(dict(root=root, before=before, after=[cell.r, cell.b]))
    prep.assumptions["student_output"] = dict(factor=2., changes=changes,
        calibration_record="memory-terminal-projection-paired-20260910/B-intact.npz",
        calibration_root="720575940628734376", calibration_tick=147,
        calibration_max_S=.5225614905357361, first_crossing_ratio=1.9136503896886632,
        derivation="Next power of two above the first student-output crossing ratio in independent novel-B records",
        limit="Birth-only excitability hypothesis, not a measured biological threshold or a guaranteed coupled-network response",
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
        feeding.run(a.graph, a.output, paired=not a.unpaired)
