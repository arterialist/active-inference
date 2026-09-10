"""Causal memory substitutions in the contact-dependent feeding preparation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .feeding import FeedingBody, step, SOMA_FIELDS, BODY_FIELDS
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from neuron.neuron import setup_neuron_logger


def probe(receiver, donor, output, *, cue="A", well=True):
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((receiver/"manifest.json").read_text())
    dm = json.loads((donor/"manifest.json").read_text())
    if cue not in m["codes"] or any(m[k] != dm[k] for k in ("graph_sha256", "factor", "motor_gain", "well_angle", "phases")):
        # The phases differ only in nutrient assignment between paired and
        # unpaired donors; their timeline and cue sequence must match.
        keys = ("graph_sha256", "factor", "motor_gain", "well_angle")
        if cue not in m["codes"] or any(m[k] != dm[k] for k in keys) or any(
                {k:v for k,v in a.items() if k!="nutrient"} != {k:v for k,v in b.items() if k!="nutrient"}
                for a,b in zip(m["phases"],dm["phases"],strict=True)):
            raise ValueError("Unmatched expression donors")
    branch = load_checkpoint(receiver/"retention.paula", trusted=True)
    source = load_checkpoint(donor/"retention.paula", trusted=True)
    with np.load(receiver/"identities.npz") as z:
        selected, ids, roots = z["selected"], z["cells"], z["roots"]
    with np.load(donor/"identities.npz") as z:
        if not np.array_equal(selected, z["selected"]):
            raise ValueError("Different memory connections")
    net = branch.network; changed = [0, 0]
    for _, pre, terminal, post, sid in selected:
        a = net.network.neurons[int(post)].postsynaptic_points[int(sid)]
        b = source.network.network.neurons[int(post)].postsynaptic_points[int(sid)]
        changed[0] += int(a.u_i.info != b.u_i.info); a.u_i.info = b.u_i.info
        a = net.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        b = source.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        changed[1] += int(a.u_o.info != b.u_o.info); a.u_o.info = b.u_o.info
    del source
    organism = FeedingBody(); organism.restore(receiver/"retention-body.npz")
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    rows = {int(nid):i for i,nid in enumerate(ids)}
    cells = [net.network.neurons[int(nid)] for nid in ids]
    soma = np.empty((200, len(cells), len(SOMA_FIELDS)))
    body = np.empty((200, len(BODY_FIELDS)))
    weights = np.empty((201, len(selected)))
    def w():
        return [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected]
    weights[0] = w()
    for t in range(200):
        body[t] = step(net, mapping, m["roles"], m["codes"], cue, organism,
                       factor=m["factor"], well=well, advance=branch.step)
        soma[t] = [[getattr(c,f) for f in SOMA_FIELDS] for c in cells]
        weights[t+1] = w()
    hits = np.flatnonzero(body[:,5] > 0)
    r = dict(receiver=receiver.name, donor=donor.name, cue=cue, well_available=well,
        changed_receiving=changed[0], changed_release=changed[1], well_j=float(body[:,5].sum()),
        first_contact_tick=int(hits[0]) if len(hits) else None, max_angle=float(body[:,0].max()),
        spikes={role: int(soma[:,[rows[mapping[root]] for root in rr],1].sum()) for role,rr in m["roles"].items()},
        weights_changed_during_probe=int(np.any(weights[1:]!=weights[0],axis=0).sum()), exact_sham=None,
        input_checkpoint_sha256=sha256(receiver/"retention.paula"), donor_checkpoint_sha256=sha256(donor/"retention.paula"),
        source_sha256=sha256(Path(__file__)))
    if receiver.resolve() == donor.resolve() and cue == "A" and well:
        phases = json.loads((receiver/"summary.json").read_text())
        p = next(x for x in phases if x["name"] == "retained_A")
        r["exact_sham"] = bool(np.array_equal(soma, np.load(receiver/"soma.npy", mmap_mode="r")[p["begin"]:p["end"]])
            and np.array_equal(body, np.load(receiver/"body.npy", mmap_mode="r")[p["begin"]:p["end"]]))
        if not r["exact_sham"]:
            raise AssertionError("Neural/physical continuation differs")
    output.mkdir(parents=True)
    np.savez_compressed(output/"trace.npz", soma=soma, body=body, weights=weights, cells=ids)
    r["trace_sha256"] = sha256(output/"trace.npz")
    (output/"summary.json").write_text(json.dumps(r,indent=2)+"\n")
    return r


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver",type=Path);p.add_argument("donor",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--cue",choices=("A","C"),default="A");p.add_argument("--dry",action="store_true")
    a=p.parse_args();print(json.dumps(probe(a.receiver,a.donor,a.output,cue=a.cue,well=not a.dry),indent=2))
