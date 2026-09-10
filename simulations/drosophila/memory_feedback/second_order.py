"""Bounded B-before-A test in the completed alpha1/gamma4 preparation.

Continue the pre-feeding retention checkpoint. Acquisition supplies no nutrient,
food well, student-DAN current, target response or learning-rate changes. A
projection-specific intervention discards only forward SMP108->student-DAN
events on arrival. All anatomical connections, reciprocal events and adaptation
remain. Separate diagnostic probes branch from the same resulting state.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .feeding import FeedingBody, step, SOMA_FIELDS, BODY_FIELDS
from ..connectome import Subgraph, sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.neuron import setup_neuron_logger


class ProjectionGate:
    def __init__(self, net, sources, targets, enabled):
        self.net = net
        self.keys = {(a, b) for a, b, c, _ in net.network.connections
                     if a in sources and c in targets}
        # Each anatomical terminal must deliver to exactly one directed pair.
        for key in self.keys:
            if sum((a, b) == key for a, b, _, _ in net.network.connections) != 1:
                raise ValueError("Projection gate would affect another target")
        if not self.keys:
            raise ValueError("Missing measured feedback projection")
        self.enabled = enabled
        self.observed = self.removed = 0

    def before_step(self):
        slot = self.net.current_tick % self.net.wheel_size
        signals = self.net.presynaptic_wheel[slot]
        kept = []
        for signal in signals:
            event = signal.event
            match = isinstance(event, tuple) and len(event) == 3 and event[:2] in self.keys
            self.observed += int(match)
            if match and self.enabled:
                self.removed += 1
            else:
                kept.append(signal)
        if self.enabled:
            self.net.presynaptic_wheel[slot] = kept


def protocol(displaced=False):
    phases = []
    for i in range(2):
        # Equal cue exposure, total duration and blank time. The temporal
        # control separates B from A by 1,000 ticks instead of 20.
        phases.extend([(f"pair{i}_B", "B", 140),
                       (f"pair{i}_gap", "", 1000 if displaced else 20),
                       (f"pair{i}_A", "A", 200),
                       (f"pair{i}_recovery", "", 20 if displaced else 1000)])
    return phases + [("retention", "", 1000)]


def run(receiver, output, *, cut=False, displaced=False):
    setup_neuron_logger("CRITICAL")
    receiver, output = Path(receiver), Path(output)
    if output.exists():
        raise FileExistsError(output)
    m = json.loads((receiver/"manifest.json").read_text())
    branch = load_checkpoint(receiver/"retention.paula", trusted=True)
    net = branch.network
    with np.load(receiver/"identities.npz") as z:
        ids, roots, selected = z["cells"], z["roots"], z["selected"]
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    rows = {int(n): i for i, n in enumerate(ids)}
    cells = [net.network.neurons[int(n)] for n in ids]
    source_ids = {mapping[r] for r in m["roles"]["SMP108"]}
    target_ids = {mapping[r] for k in ("PAM07", "PAM08") for r in m["roles"][k]}
    gate = ProjectionGate(net, source_ids, target_ids, cut)
    graph = Subgraph.load(Path(m["graph"]))
    projection = [dict(source=str(e[0]), target=str(e[1]), source_row=int(e[8]), count=int(e[4]))
                  for e in graph.internal if int(e[2]) in source_ids and int(e[3]) in target_ids]
    organism = FeedingBody(); organism.restore(receiver/"retention-body.npz")
    phases = protocol(displaced); total = sum(p[2] for p in phases)
    output.mkdir(parents=True)
    def array(name, shape):
        return np.lib.format.open_memmap(output/(name+".npy"), mode="w+", dtype="float64", shape=shape)
    soma = array("soma", (total, len(ids), len(SOMA_FIELDS)))
    mods = array("modulation", (total, len(ids), 2))
    body = array("body", (total, len(BODY_FIELDS)))
    weights = array("weights", (total+1, len(selected)))
    release = array("release", (total+1, len(selected)))
    def coefficients(network):
        return ([network.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected],
                [network.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected])
    weights[0], release[0] = coefficients(net)
    reports = []; t = 0
    for name, cue, ticks in phases:
        begin = t
        for _ in range(ticks):
            gate.before_step()
            body[t] = step(net, mapping, m["roles"], m["codes"], cue, organism,
                           factor=m["factor"], advance=branch.step)
            soma[t] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cells]
            mods[t] = [c.M_vector for c in cells]
            weights[t+1], release[t+1] = coefficients(net)
            if not all(np.isfinite(x).all() for x in (soma[t], mods[t], body[t], weights[t+1], release[t+1])):
                raise FloatingPointError("Nonfinite continuation")
            t += 1
        report = dict(name=name, begin=begin, end=t, cue=cue,
            spikes={role: int(soma[begin:t, [rows[mapping[r]] for r in rr], 1].sum()) for role, rr in m["roles"].items()},
            max_angle=float(body[begin:t, 0].max()), feedback_events=gate.observed, removed_events=gate.removed)
        reports.append(report); print(json.dumps(report), flush=True)
    if np.any(body[:, 4:6]):
        raise AssertionError("Second-order acquisition must have no nutrients")
    save_checkpoint(net, output/"retention.paula", sources=[__file__])
    organism.save(output/"retention-body.npz")
    for a in (soma, mods, body, weights, release):
        a.flush()
    np.savez_compressed(output/"identities.npz", cells=ids, roots=roots, selected=selected)
    # No intervention during expression. Both probes start before either probe
    # can change the retained preparation; adaptation continues within each.
    probes = {}
    for cue, well in (("B", False), ("A", True)):
        b = load_checkpoint(output/"retention.paula", trusted=True)
        o = FeedingBody(); o.restore(output/"retention-body.npz")
        ss = np.empty((200, len(ids), len(SOMA_FIELDS)))
        mm = np.empty((200, len(ids), 2)); bb = np.empty((200, len(BODY_FIELDS)))
        ww = np.empty((201, len(selected))); rr = np.empty_like(ww)
        ww[0], rr[0] = coefficients(b.network)
        cc = [b.network.network.neurons[int(n)] for n in ids]
        for j in range(200):
            bb[j] = step(b.network, mapping, m["roles"], m["codes"], cue, o,
                         factor=m["factor"], well=well, advance=b.step)
            ss[j] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cc]
            mm[j] = [c.M_vector for c in cc]
            ww[j+1], rr[j+1] = coefficients(b.network)
        np.savez_compressed(output/("probe-"+cue+".npz"), soma=ss, modulation=mm, body=bb, weights=ww, release=rr)
        probes[cue] = dict(well_available=well, well_j=float(bb[:, 5].sum()), max_angle=float(bb[:, 0].max()),
            changed_weights=int(np.any(ww[1:] != ww[0], axis=0).sum()),
            spikes={role: int(ss[:, [rows[mapping[r]] for r in roots_], 1].sum()) for role, roots_ in m["roles"].items()})
    result = dict(receiver=receiver.name, cut=cut, displaced=displaced,
        input_checkpoint_sha256=sha256(receiver/"retention.paula"),
        source_sha256=sha256(Path(__file__)), graph_sha256=m["graph_sha256"],
        phases=reports, probes=probes, projection=projection,
        feedback_events=gate.observed, removed_events=gate.removed,
        nutrient_j=float(body[:, 4:6].sum()),
        minimum_eta_post=min(c.params.eta_post for c in cells),
        minimum_eta_retro=min(c.params.eta_retro for c in cells),
        limit="Two engineered cue pairings in a student compartment already found silent. No positive second-order claim from exposure-driven weight changes.")
    result["artifacts"] = {f.name: sha256(f) for f in output.iterdir() if f.is_file()}
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--cut", action="store_true"); p.add_argument("--displaced", action="store_true")
    a = p.parse_args(); run(a.receiver, a.output, cut=a.cut, displaced=a.displaced)
