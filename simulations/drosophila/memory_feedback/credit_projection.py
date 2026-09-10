"""Probe a fixed fourfold log-depression projection before longer training.

This is an imposed coefficient diagnostic. A 32-pairing course need not follow
the projection because neural feedback, inhibition and native adaptation can
change during acquisition. The multiplier is fixed before this probe.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import interface_controls as controls
from ..connectome import sha256


def run(parent, intact, cut, output):
    parent, intact, cut, output = map(Path, (parent, intact, cut, output))
    def intervention(branch, donor, selected, mask):
        changes, originals = [], []
        for row, pre, terminal, post, sid in selected[mask]:
            point = branch.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
            current = float(point.u_o.info)
            baseline = float(donor.network.network.neurons[int(pre)].presynaptic_points[int(terminal)].u_o.info)
            if baseline-current <= 1e-6: continue
            if not 0 < current < baseline: raise ValueError("Invalid depressive contrast")
            value = baseline*(current/baseline)**4
            originals.append((point, point.u_o.info))
            changes.append(dict(source_row=int(row), before=current, cut=baseline, after=value))
            point.u_o.info = value
        return changes, originals
    with patch.object(controls, "swap", intervention):
        result = controls.probe(parent, parent, output, state=intact, donor_state=cut,
                                cue="B", well=False, student_only=True)
    result["projection"] = dict(log_depression_multiplier=4, source_sha256=sha256(Path(__file__)),
        limit="Imposed 32/8 exposure extrapolation at the eight-pairing retained state. It is not learned memory or a prediction of the full coupled course with changing neural activity.")
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "intact", "cut", "output"): p.add_argument(name,type=Path)
    a=p.parse_args();run(a.parent,a.intact,a.cut,a.output)
