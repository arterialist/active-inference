"""Test whether recruited student output suppresses the retained A teacher.

Cut only MBON04-to-SMP108 forward events during independent dry A probes.
The paired and unpaired first-order parents otherwise retain their full state
and all adaptation. This diagnoses a feedback conflict, not B acquisition.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .feeding import FeedingBody, step, SOMA_FIELDS, BODY_FIELDS
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
        ids, roots = z["cells"], z["roots"]
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    rows = dict(zip(roots.tolist(), range(len(roots)), strict=True))
    output.mkdir(parents=True)
    result = dict(receiver=receiver.name, checkpoint_sha256=sha256(receiver/"retention.paula"),
        source_sha256=sha256(Path(__file__)), probes={})
    for cut in (False, True):
        branch = load_checkpoint(receiver/"retention.paula", trusted=True)
        net = branch.network
        body = FeedingBody(); body.restore(receiver/"retention-body.npz")
        gate = ProjectionGate(net, {mapping[r] for r in m["roles"]["MBON04"]},
                              {mapping[r] for r in m["roles"]["SMP108"]}, cut)
        soma = np.empty((200, len(ids), len(SOMA_FIELDS)))
        physical = np.empty((200, len(BODY_FIELDS)))
        cells = [net.network.neurons[int(n)] for n in ids]
        for t in range(200):
            gate.before_step()
            physical[t] = step(net, mapping, m["roles"], m["codes"], "A", body,
                               factor=m["factor"], advance=branch.step)
            soma[t] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cells]
        assert not np.any(physical[:, 4:6])
        name = "cut" if cut else "intact"
        np.savez_compressed(output/(name+".npz"), soma=soma, body=physical)
        result["probes"][name] = dict(removed_events=gate.removed, observed_events=gate.observed,
            max_angle=float(physical[:, 0].max()), nutrient_j=float(physical[:, 4:6].sum()),
            spikes={role: int(soma[:, [rows[r] for r in rr], 1].sum()) for role, rr in m["roles"].items()},
            trace_sha256=sha256(output/(name+".npz")))
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.receiver, a.output)
