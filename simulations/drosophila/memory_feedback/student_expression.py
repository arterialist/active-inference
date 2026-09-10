"""Diagnose expression through only the gamma4 B terminal coefficients.

Independent dry B branches either retain their memory, substitute coefficients
from a matched checkpoint, or set those coefficients to zero as a maximum-
depression diagnostic. Zeroing is an intervention, never a learning rule or
evidence that the intact preparation acquired that state. All other state and
ongoing adaptation remain. The selected motor boundary is unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import feeding, primary_boundary
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def run(parent, state, output, *, donor=None, zero=False):
    setup_neuron_logger("CRITICAL")
    parent, state, output = map(Path, (parent, state, output))
    if output.exists():
        raise FileExistsError(output)
    if donor is not None and zero:
        raise ValueError("Choose one coefficient intervention")
    m = json.loads((parent/"manifest.json").read_text())
    with np.load(parent/"identities.npz") as z:
        ids, roots, selected = z["cells"], z["roots"], z["selected"]
    def verify(path):
        with np.load(path/"identities.npz") as z:
            if not all(np.array_equal(a, z[k]) for k, a in (("cells", ids), ("roots", roots), ("selected", selected))):
                raise ValueError("Mismatched expression preparation")
        if path.resolve() != parent.resolve():
            summary = json.loads((path/"summary.json").read_text())
            if summary["input_checkpoint_sha256"] != sha256(parent/"retention.paula"):
                raise ValueError("State does not descend from the declared parent")
    verify(state)
    branch = load_checkpoint(state/"retention.paula", trusted=True)
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    rows = dict(zip(roots.tolist(), range(len(roots)), strict=True))
    mask = np.isin(selected[:, 1], [mapping[r] for r in m["codes"]["B"]]) & np.isin(
        selected[:, 3], [mapping[r] for r in m["roles"]["MBON04"]])
    changes = []
    source = None
    if donor is not None:
        donor = Path(donor); verify(donor)
        source = load_checkpoint(donor/"retention.paula", trusted=True)
    for row, pre, terminal, post, sid in selected[mask]:
        point = branch.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        value = (0. if zero else source.network.network.neurons[int(pre)].presynaptic_points[int(terminal)].u_o.info
                 if source is not None else point.u_o.info)
        changes.append(dict(source_row=int(row), before=float(point.u_o.info), after=float(value)))
        point.u_o.info = value
    del source
    organism = feeding.FeedingBody(); organism.restore(state/"retention-body.npz")
    step = primary_boundary.step if "primary_boundary" in m["assumptions"] else feeding.step
    ss = np.empty((200, len(ids), len(feeding.SOMA_FIELDS)))
    bb = np.empty((200, len(feeding.BODY_FIELDS)))
    qq = np.empty((201, len(selected)))
    ww = np.empty_like(qq)
    net = branch.network
    cells = [net.network.neurons[int(n)] for n in ids]
    def coefficients():
        return ([net.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected],
                [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
    qq[0], ww[0] = coefficients()
    for t in range(200):
        bb[t] = step(net, mapping, m["roles"], m["codes"], "B", organism,
                     factor=m["factor"], advance=branch.step)
        ss[t] = [[getattr(c, f) for f in feeding.SOMA_FIELDS] for c in cells]
        qq[t+1], ww[t+1] = coefficients()
    assert not np.any(bb[:, 4:6])
    output.mkdir(parents=True)
    np.savez_compressed(output/"trace.npz", soma=ss, body=bb, release=qq, weights=ww)
    result = dict(parent=parent.name, state=state.name, donor=donor.name if donor else None, zero=zero,
        changed_coefficients=changes, changed_receiving_by_intervention=0,
        source_sha256=sha256(Path(__file__)), parent_manifest_sha256=sha256(parent/"manifest.json"),
        state_checkpoint_sha256=sha256(state/"retention.paula"),
        donor_checkpoint_sha256=sha256(donor/"retention.paula") if donor else None,
        trace_sha256=sha256(output/"trace.npz"), nutrient_j=float(bb[:, 4:6].sum()),
        max_angle=float(bb[:, 0].max()),
        spikes={role: int(ss[:, [rows[r] for r in rr], 1].sum()) for role, rr in m["roles"].items()},
        limit="Dry diagnostic with continuing adaptation. Zero-release results are an imposed expression bound, not acquired memory.")
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path); p.add_argument("state", type=Path); p.add_argument("output", type=Path)
    g=p.add_mutually_exclusive_group(); g.add_argument("--donor", type=Path); g.add_argument("--zero", action="store_true")
    a=p.parse_args(); run(a.parent, a.state, a.output, donor=a.donor, zero=a.zero)
