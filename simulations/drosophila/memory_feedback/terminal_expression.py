"""Swap only stored KC terminal release coefficients, retaining all other state."""
import argparse
import json
from pathlib import Path

import numpy as np

from .feeding import FeedingBody, step, SOMA_FIELDS
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def probe(receiver, donor, output, *, cue="A", well=True):
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    if output.exists():
        raise FileExistsError(output)
    m, dm = [json.loads((p/"manifest.json").read_text()) for p in (receiver, donor)]
    if any(m[k] != dm[k] for k in ("graph_sha256", "factor", "motor_gain", "well_angle", "codes")):
        raise ValueError("Unmatched terminal-memory donors")
    branch = load_checkpoint(receiver/"retention.paula", trusted=True)
    source = load_checkpoint(donor/"retention.paula", trusted=True)
    with np.load(receiver/"identities.npz") as z:
        selected, ids, roots = z["selected"], z["cells"], z["roots"]
    with np.load(donor/"identities.npz") as z:
        if not np.array_equal(selected, z["selected"]):
            raise ValueError("Different memory terminals")
    net = branch.network; changed = 0
    for _, pre, terminal, _, _ in selected:
        a = net.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        b = source.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        changed += int(a.u_o.info != b.u_o.info)
        a.u_o.info = b.u_o.info
    del source
    organism = FeedingBody(); organism.restore(receiver/"retention-body.npz")
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True)); rows = {int(n): i for i, n in enumerate(ids)}
    cells = [net.network.neurons[int(n)] for n in ids]
    locations = [(c.id, j) for c in cells for j, _ in enumerate(getattr(c, "terminal_credit_groups", ()))]
    soma = np.empty((200, len(ids), len(SOMA_FIELDS)))
    body = np.empty((200, 10)); chemistry = np.empty((200, len(locations), 2))
    release = np.empty((201, len(selected))); weights = np.empty_like(release)
    def coefficients():
        return ([net.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected],
                [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
    release[0], weights[0] = coefficients()
    for t in range(200):
        body[t] = step(net, mapping, m["roles"], m["codes"], cue, organism,
                       factor=m["factor"], well=well, advance=branch.step)
        soma[t] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cells]
        release[t+1], weights[t+1] = coefficients()
        chemistry[t] = [[net.network.neurons[n].terminal_credit_kc,
                         net.network.neurons[n].terminal_credit_groups[j]["dopamine"]] for n, j in locations]
    r = dict(receiver=receiver.name, donor=donor.name, cue=cue, well_available=well,
        changed_release=changed, changed_receiving_by_intervention=0,
        well_j=float(body[:, 5].sum()), max_angle=float(body[:, 0].max()),
        spikes={role: int(soma[:, [rows[mapping[root]] for root in rr], 1].sum()) for role, rr in m["roles"].items()},
        changed_release_during_probe=int(np.any(release[1:] != release[0], axis=0).sum()),
        changed_receiving_during_probe=int(np.any(weights[1:] != weights[0], axis=0).sum()),
        exact_sham=None, input_checkpoint_sha256=sha256(receiver/"retention.paula"),
        donor_checkpoint_sha256=sha256(donor/"retention.paula"), source_sha256=sha256(Path(__file__)))
    if receiver.resolve() == donor.resolve() and cue == "A" and well:
        phase = next(x for x in json.loads((receiver/"summary.json").read_text()) if x["name"] == "retained_A")
        r["exact_sham"] = bool(np.array_equal(soma, np.load(receiver/"soma.npy", mmap_mode="r")[phase["begin"]:phase["end"]]) and
            np.array_equal(body, np.load(receiver/"body.npy", mmap_mode="r")[phase["begin"]:phase["end"]]))
        if not r["exact_sham"]:
            raise AssertionError("Continuation differs from recorded somatic/physical course")
    output.mkdir(parents=True)
    np.savez_compressed(output/"trace.npz", soma=soma, body=body, chemistry=chemistry,
                        group_locations=np.array(locations), release=release, weights=weights)
    r["trace_sha256"] = sha256(output/"trace.npz")
    (output/"summary.json").write_text(json.dumps(r, indent=2)+"\n")
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("donor", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--cue", choices=("A", "B", "C"), default="A"); p.add_argument("--dry", action="store_true")
    a=p.parse_args(); print(json.dumps(probe(a.receiver,a.donor,a.output,cue=a.cue,well=not a.dry),indent=2))
