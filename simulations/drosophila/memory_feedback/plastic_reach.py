"""Impose an expression bound only at B terminals changed by feedback.

The intact-minus-cut coefficient contrast identifies reached terminals after
the fixed eight-pairing comparison. Zeroing those terminals is a diagnostic,
not a learning rule or a claim that further exposure reaches this state.
"""
import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import interface_controls as controls
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import _serializer


def reached_mask(selected_mask, intact, cut, tolerance=1e-6):
    return selected_mask & (np.asarray(cut)-np.asarray(intact) > tolerance)


def run(parent, intact, cut, output, *, mode="reached"):
    parent, intact, cut, output = map(Path, (parent, intact, cut, output))
    if mode not in ("sham", "reached", "all"):
        raise ValueError("Unknown expression bound")
    receipts = {}

    def intervention(branch, donor, selected, mask):
        points = [branch.network.network.neurons[int(e[1])].presynaptic_points[int(e[2])] for e in selected]
        q = np.array([p.u_o.info for p in points])
        dq = np.array([donor.network.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected])
        reached = reached_mask(mask, q, dq)
        chosen = reached if mode == "reached" else mask if mode == "all" else np.zeros_like(mask)
        _, pickler = _serializer()
        def blob():
            buf = io.BytesIO()
            pickler(buf, protocol=5).dump(dict(network=branch.network, python_rng=branch.python_rng, numpy_rng=branch.numpy_rng))
            return buf.getvalue()
        before = blob()
        originals = [(points[i], points[i].u_o.info) for i in np.flatnonzero(chosen)]
        changes = [dict(source_row=int(selected[i, 0]), before=float(q[i]), after=0.) for i in np.flatnonzero(chosen)]
        for p, value in originals: p.u_o.info = 0.
        for p, value in originals: p.u_o.info = value
        receipts["all_other_runtime_state_exact"] = before == blob()
        assert receipts["all_other_runtime_state_exact"]
        for p, value in originals: p.u_o.info = 0.
        receipts.update(mode=mode, reached_count=int(reached.sum()), candidate_count=int(mask.sum()),
                        reached_source_rows=selected[reached, 0].tolist(), tolerance=1e-6)
        return changes, originals

    with patch.object(controls, "swap", intervention):
        result = controls.probe(parent, parent, output, state=intact, donor_state=cut,
                                cue="B", well=False, student_only=True)
    result["expression_bound"] = receipts
    result["bound_driver_sha256"] = sha256(Path(__file__))
    result["limit"] = "Imposed zero release at selected terminals with all other initial state preserved and adaptation continuing. This is not acquired behavior or proof that repeated training reaches the imposed state."
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "intact", "cut", "output"): p.add_argument(name, type=Path)
    p.add_argument("--mode", choices=("sham", "reached", "all"), default="reached")
    a=p.parse_args();run(a.parent,a.intact,a.cut,a.output,mode=a.mode)
