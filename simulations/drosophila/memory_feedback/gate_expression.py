"""Validate feedback interruption against retained A expression before B training."""
import argparse
import json
from pathlib import Path

import numpy as np

from .second_order import ProjectionGate
from .feeding import FeedingBody, step, SOMA_FIELDS
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
    output.mkdir(parents=True)
    traces = []; reports = []
    for cut in (False, True):
        b = load_checkpoint(receiver/"retention.paula", trusted=True)
        net = b.network
        o = FeedingBody(); o.restore(receiver/"retention-body.npz")
        gate = ProjectionGate(net, {mapping[r] for r in m["roles"]["SMP108"]},
            {mapping[r] for k in ("PAM07", "PAM08") for r in m["roles"][k]}, cut)
        cells = [net.network.neurons[int(n)] for n in ids]
        soma = []; body = []; weights = []
        for _ in range(200):
            gate.before_step()
            body.append(step(net, mapping, m["roles"], m["codes"], "A", o,
                             factor=m["factor"], well=True, advance=b.step))
            soma.append([[getattr(c, f) for f in SOMA_FIELDS] for c in cells])
            weights.append([net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
        soma, body, weights = map(np.asarray, (soma, body, weights))
        traces.append((soma, body, weights))
        np.savez_compressed(output/("cut.npz" if cut else "intact.npz"), soma=soma, body=body, weights=weights)
        reports.append(dict(cut=cut, removed_events=gate.removed, observed_events=gate.observed,
            well_j=float(body[:, 5].sum()), max_angle=float(body[:, 0].max()),
            spikes={role: int(soma[:, [rows[mapping[r]] for r in rr], 1].sum()) for role, rr in m["roles"].items()}))
    teacher = [rows[mapping[r]] for role in ("MBON07", "SMP353", "SMP108") for r in m["roles"][role]]
    phase = next(p for p in json.loads((receiver/"summary.json").read_text()) if p["name"] == "retained_A")
    result = dict(receiver=receiver.name, branches=reports,
        original_course_replay=bool(np.array_equal(traces[0][0], np.load(receiver/"soma.npy", mmap_mode="r")[phase["begin"]:phase["end"]])
            and np.array_equal(traces[0][1], np.load(receiver/"body.npy", mmap_mode="r")[phase["begin"]:phase["end"]])),
        exact_teacher_motor_soma=bool(np.array_equal(traces[0][0][:, teacher], traces[1][0][:, teacher])),
        exact_body=bool(np.array_equal(traces[0][1], traces[1][1])),
        exact_selected_weights=bool(np.array_equal(traces[0][2], traces[1][2])),
        source_sha256=sha256(Path(__file__)), input_checkpoint_sha256=sha256(receiver/"retention.paula"),
        traces={f: sha256(output/f) for f in ("cut.npz", "intact.npz")})
    if not all(result[k] for k in ("original_course_replay", "exact_teacher_motor_soma", "exact_body", "exact_selected_weights")):
        raise AssertionError("Feedback interruption changes retained A expression")
    if reports[1]["removed_events"] == 0:
        raise AssertionError("Inactive intervention")
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); print(json.dumps(run(a.receiver, a.output), indent=2))
