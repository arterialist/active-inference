"""Dry cue probes separate student recruitment from candidate feedback dependence."""
import argparse
import json
from pathlib import Path

import numpy as np

from .feeding import FeedingBody, step, SOMA_FIELDS
from .second_order import ProjectionGate
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def run(receiver, output):
    setup_neuron_logger("CRITICAL")
    receiver, output = Path(receiver), Path(output)
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((receiver/"manifest.json").read_text())
    with np.load(receiver/"identities.npz") as z:
        ids, roots, selected = z["cells"], z["roots"], z["selected"]
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    rows = {int(n): i for i, n in enumerate(ids)}
    output.mkdir(parents=True); reports = {}
    for cue in ("A", "B", "C"):
        for cut in (False, True):
            branch = load_checkpoint(receiver/"retention.paula", trusted=True)
            net = branch.network
            body = FeedingBody(); body.restore(receiver/"retention-body.npz")
            gate = ProjectionGate(net, {mapping[r] for r in m["roles"]["SMP108"]},
                {mapping[r] for role in ("PAM07", "PAM08") for r in m["roles"][role]}, cut)
            cells = [net.network.neurons[int(n)] for n in ids]
            ss, bb, mm, ww = [], [], [], []
            for _ in range(200):
                gate.before_step()
                bb.append(step(net, mapping, m["roles"], m["codes"], cue, body,
                    factor=m["factor"], advance=branch.step))
                ss.append([[getattr(c, f) for f in SOMA_FIELDS] for c in cells])
                mm.append([c.M_vector.copy() for c in cells])
                ww.append([net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
            ss, bb, mm, ww = map(np.asarray, (ss, bb, mm, ww))
            name = cue+("-cut" if cut else "-intact")
            np.savez_compressed(output/(name+".npz"), soma=ss, body=bb, modulation=mm, weights=ww)
            reports[name] = dict(cue=cue, cut=cut, observed_events=gate.observed, removed_events=gate.removed,
                nutrient_j=float(bb[:, 4:6].sum()), max_angle=float(bb[:, 0].max()),
                roles={role: dict(spikes=int(ss[:, [rows[mapping[r]] for r in rr], 1].sum()),
                    max_S=float(ss[:, [rows[mapping[r]] for r in rr], 0].max()),
                    max_M1=float(mm[:, [rows[mapping[r]] for r in rr], 1].max())) for role, rr in m["roles"].items()},
                trace_sha256=sha256(output/(name+".npz")))
            assert reports[name]["nutrient_j"] == 0.
            print(json.dumps(reports[name]), flush=True)
    result = dict(receiver=receiver.name, input_checkpoint_sha256=sha256(receiver/"retention.paula"),
        source_sha256=sha256(Path(__file__)), probes=reports,
        limit="Independent diagnostic branches with continuing adaptation. B is novel; recruitment by B alone cannot establish memory-mediated teaching.")
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); run(a.receiver, a.output)
