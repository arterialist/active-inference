"""Swap only stored KC->MBON coefficients between matched acquisition histories.

These are diagnostic branches, not resets imposed during ordinary acquisition.
All other neural state, queued events, organs and body state belong to the
receiver. Both learning directions remain enabled during each 200-tick probe.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .acquisition import SOMA_FIELDS, PHYSICAL_FIELDS
from ...active_inference.core.runtime_checkpoint import load_checkpoint
from ...active_inference.components.body.loaded_hinge import LoadedHinge, DT, MOTOR_GEAR
from ...active_inference.components.body.energy_budget import EnergyBudget, EnergyBudgetParameters
from ..connectome import sha256


def probe(receiver, donor, output, cue="A"):
    from neuron.neuron import setup_neuron_logger
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((receiver / "manifest.json").read_text())
    dm = json.loads((donor / "manifest.json").read_text())
    if cue not in manifest["codes"] or any(manifest[k] != dm[k]
            for k in ("graph_sha256", "rule", "blocks", "protocol_version")):
        raise ValueError("Unmatched acquisition histories or unknown cue")
    branch = load_checkpoint(receiver / "retention.paula", trusted=True)
    source = load_checkpoint(donor / "retention.paula", trusted=True)
    net = branch.network
    with np.load(receiver / "identities.npz", allow_pickle=False) as z:
        selected, ids, roots = z["selected"], z["cells"], z["roots"]
    with np.load(donor / "identities.npz", allow_pickle=False) as z:
        if not np.array_equal(selected, z["selected"]):
            raise ValueError("Synaptic identities differ")
    changed_receiving = changed_release = 0
    for _, pre, terminal, post, sid in selected:
        p = net.network.neurons[int(post)].postsynaptic_points[int(sid)]
        d = source.network.network.neurons[int(post)].postsynaptic_points[int(sid)]
        changed_receiving += int(p.u_i.info != d.u_i.info)
        p.u_i.info = d.u_i.info
        p = net.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        d = source.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        changed_release += int(p.u_o.info != d.u_o.info)
        p.u_o.info = d.u_o.info
    del source
    body = LoadedHinge(spring=.15)
    organs = EnergyBudget(energy_j=0., gut_j=0.,
                          params=EnergyBudgetParameters(**manifest["energy_parameters"]))
    with np.load(receiver / "retention-body.npz", allow_pickle=False) as z:
        body.restore(z["state"])
        for name, value in zip(organs.fields, z["energy"], strict=True):
            setattr(organs, name, float(value))
        muscle = float(z["muscle"])
    root_to_id = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    roles = manifest["roles"]
    active = set(manifest["codes"][cue])
    cells = [net.network.neurons[int(i)] for i in ids]
    ticks = 200
    soma = np.empty((ticks, len(cells), len(SOMA_FIELDS)))
    physical = np.empty((ticks, len(PHYSICAL_FIELDS)))
    weights = np.empty((ticks+1, len(selected)))
    weights[0] = [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected]
    for t in range(ticks):
        for code in manifest["codes"].values():
            for root in code:
                cell = net.network.neurons[root_to_id[root]]
                net.set_external_input(cell.id, cell.params.num_inputs-1, 40. if root in active else 0.)
        for root in roles["PAM11"]:
            cell = net.network.neurons[root_to_id[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 0.)
        for root in roles["SMP353"]:
            cell = net.network.neurons[root_to_id[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 1.4)
        branch.step()
        muscle = .9*muscle + .1*np.mean([net.network.neurons[root_to_id[r]].O for r in roles["SMP353"]])
        angle = float(body.data.qpos[0])
        body.step(muscle)
        work = max(0., MOTOR_GEAR*muscle*(float(body.data.qpos[0])-angle))
        digested, _, _, _ = organs.advance(DT, work, muscle*muscle*DT)
        soma[t] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cells]
        physical[t] = [body.data.qpos[0], body.data.qvel[0], muscle, 0., digested,
                       organs.energy_j, organs.gut_j, organs.unmet_j]
        weights[t+1] = [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected]
    rows = {int(nid): i for i, nid in enumerate(ids)}
    report = dict(receiver=str(receiver.resolve()), donor=str(donor.resolve()), cue=cue,
        changed_receiving=changed_receiving, changed_release=changed_release,
        spikes={role: int(soma[:, [rows[root_to_id[r]] for r in rr], 1].sum()) for role, rr in roles.items()},
        mean_angle=float(physical[:, 0].mean()),
        changed_weights_during_probe=int(np.any(weights[1:] != weights[0], axis=0).sum()),
        exact_A_sham_replay=None,
        sources={str(p.resolve()): sha256(p) for p in
                 (receiver/"retention.paula", donor/"retention.paula", Path(__file__))})
    if receiver.resolve() == donor.resolve() and cue == "A":
        phases = json.loads((receiver / "summary.json").read_text())
        window = next(p for p in phases if p["name"] == "retained_A")
        original_soma = np.load(receiver / "soma.npy", mmap_mode="r")
        original_physical = np.load(receiver / "physical.npy", mmap_mode="r")
        report["exact_A_sham_replay"] = bool(
            np.array_equal(soma, original_soma[window["begin"]:window["end"]]) and
            np.array_equal(physical, original_physical[window["begin"]:window["end"]]))
        if not report["exact_A_sham_replay"]:
            raise AssertionError("Restored neural/physical continuation differs")
    output.mkdir(parents=True)
    np.savez_compressed(output / "trace.npz", soma=soma, physical=physical, weights=weights, cells=ids)
    report["trace_sha256"] = sha256(output / "trace.npz")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receiver", type=Path); p.add_argument("donor", type=Path)
    p.add_argument("output", type=Path); p.add_argument("--cue", choices=("A", "C"), default="A")
    a = p.parse_args()
    print(json.dumps(probe(a.receiver, a.donor, a.output, a.cue), indent=2))
