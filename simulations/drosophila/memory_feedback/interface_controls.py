"""Cue-specific terminal-memory controls for the student-interface assay."""
import argparse
import io
import json
from pathlib import Path
import shutil

import numpy as np

from . import feeding, student_interface
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, save_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def read(path):
    return json.loads(Path(path).read_text())


def pair(receiver, donor):
    m, dm = [read(p/"manifest.json") for p in (receiver, donor)]
    for key in ("graph_sha256", "codes", "roles", "factor", "motor_gain", "well_angle", "assumptions"):
        if m[key] != dm[key]:
            raise ValueError("Unmatched memory donors: "+key)
    if m["assumptions"]["student_interface"]["source_sha256"] != sha256(Path(student_interface.__file__)):
        raise ValueError("Student interface differs from acquisition")
    with np.load(receiver/"identities.npz") as z:
        ids, roots, selected = z["cells"], z["roots"], z["selected"]
    with np.load(donor/"identities.npz") as z:
        assert all(np.array_equal(a, z[k]) for k, a in (("cells", ids), ("roots", roots), ("selected", selected)))
    return m, ids, roots, selected


def mask_for(m, ids, roots, selected, cue, targets):
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
    return np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) & np.isin(
        selected[:, 3], [mapping[r] for role in targets for r in m["roles"][role]])


def swap(branch, donor, selected, mask):
    changes, originals = [], []
    for row, pre, terminal, post, sid in selected[mask]:
        point = branch.network.network.neurons[int(pre)].presynaptic_points[int(terminal)]
        value = donor.network.network.neurons[int(pre)].presynaptic_points[int(terminal)].u_o.info
        originals.append((point, point.u_o.info))
        changes.append(dict(source_row=int(row), before=float(point.u_o.info), after=float(value)))
        point.u_o.info = value
    return changes, originals


def probe(receiver, donor, output, *, state=None, donor_state=None, cue="A", well=True, student_only=False):
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    state, donor_state = Path(state or receiver), Path(donor_state or donor)
    if output.exists(): raise FileExistsError(output)
    m, ids, roots, selected = pair(receiver, donor)
    for p, parent in ((state, receiver), (donor_state, donor)):
        if p.resolve() != parent.resolve() and read(p/"summary.json")["input_checkpoint_sha256"] != sha256(parent/"retention.paula"):
            raise ValueError("State does not descend from its declared parent")
    branch = load_checkpoint(state/"retention.paula", trusted=True)
    source = load_checkpoint(donor_state/"retention.paula", trusted=True)
    targets = ("MBON04",) if student_only else ("MBON07", "MBON04")
    mask = mask_for(m, ids, roots, selected, cue, targets)
    changes, _ = swap(branch, source, selected, mask); del source
    net = branch.network
    mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True)); rows = dict(zip(roots.tolist(), range(len(roots))))
    organism = feeding.FeedingBody(); organism.restore(state/"retention-body.npz")
    cells = [net.network.neurons[int(n)] for n in ids]
    ss = np.empty((200, len(ids), len(feeding.SOMA_FIELDS))); bb = np.empty((200, len(feeding.BODY_FIELDS)))
    qq = np.empty((201, len(selected))); ww = np.empty_like(qq)
    def coefficients():
        return ([net.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected],
                [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected])
    qq[0], ww[0] = coefficients()
    for t in range(200):
        bb[t] = student_interface.step(net, mapping, m["roles"], m["codes"], cue, organism,
                                      factor=m["factor"], well=well, advance=branch.step)
        ss[t] = [[getattr(c, f) for f in feeding.SOMA_FIELDS] for c in cells]
        qq[t+1], ww[t+1] = coefficients()
    if not well: assert not np.any(bb[:, 4:6])
    exact = None
    if state.resolve() == donor_state.resolve() and state.resolve() == receiver.resolve() and cue == "A" and well:
        ph = next(x for x in read(receiver/"summary.json") if x["name"] == "retained_A")
        exact = bool(np.array_equal(ss, np.load(receiver/"soma.npy", mmap_mode="r")[ph["begin"]:ph["end"]]) and
            np.array_equal(bb, np.load(receiver/"body.npy", mmap_mode="r")[ph["begin"]:ph["end"]]))
        assert exact
    output.mkdir(parents=True)
    np.savez_compressed(output/"trace.npz", soma=ss, body=bb, release=qq, weights=ww)
    result = dict(receiver=receiver.name, state=state.name, donor=donor.name, donor_state=donor_state.name,
        cue=cue, well=well, targets=targets, changes=changes, changed_receiving_by_intervention=0,
        source_sha256=sha256(Path(__file__)), input_checkpoint_sha256=sha256(state/"retention.paula"),
        donor_checkpoint_sha256=sha256(donor_state/"retention.paula"), trace_sha256=sha256(output/"trace.npz"),
        exact_sham=exact, well_j=float(bb[:, 5].sum()), max_angle=float(bb[:, 0].max()),
        first_contact_tick=int(np.flatnonzero(bb[:, 0] >= feeding.WELL_ANGLE)[0]) if np.any(bb[:, 0] >= feeding.WELL_ANGLE) else None,
        spikes={role: int(ss[:, [rows[r] for r in rr], 1].sum()) for role, rr in m["roles"].items()})
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def teacher(receiver, donor, output, *, alpha_only=False):
    setup_neuron_logger("CRITICAL")
    receiver, donor, output = map(Path, (receiver, donor, output))
    if output.exists(): raise FileExistsError(output)
    m, ids, roots, selected = pair(receiver, donor)
    branch = load_checkpoint(receiver/"retention.paula", trusted=True)
    source = load_checkpoint(donor/"retention.paula", trusted=True)
    _, pickler = _serializer()
    def blob():
        buf = io.BytesIO(); pickler(buf, protocol=5).dump(dict(network=branch.network, python_rng=branch.python_rng, numpy_rng=branch.numpy_rng))
        return buf.getvalue()
    before = blob()
    targets = ("MBON07",) if alpha_only else ("MBON07", "MBON04")
    changes, originals = swap(branch, source, selected, mask_for(m, ids, roots, selected, "A", targets))
    del source
    target = output/"receiver"; target.mkdir(parents=True)
    receipt = dict(receiver=receiver.name, donor=donor.name, cue="A", targets=targets, changes=changes,
        source_sha256=sha256(Path(__file__)), input_checkpoint_sha256=sha256(receiver/"retention.paula"),
        donor_checkpoint_sha256=sha256(donor/"retention.paula"))
    m["cue_memory_substitution"] = receipt
    (target/"manifest.json").write_text(json.dumps(m, indent=2)+"\n")
    for f in ("identities.npz", "retention-body.npz"): shutil.copyfile(receiver/f, target/f)
    save_checkpoint(branch, target/"retention.paula", sources=[__file__])
    for point, value in originals: point.u_o.info = value
    receipt["all_other_runtime_state_exact"] = blob() == before
    assert receipt["all_other_runtime_state_exact"]
    del branch, before, originals
    result = student_interface.run_continuation(target, output/"continuation")
    receipt["substituted_checkpoint_sha256"] = result["input_checkpoint_sha256"]
    receipt["continuation_summary_sha256"] = sha256(output/"continuation"/"summary.json")
    (output/"intervention.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("probe", "teacher"))
    p.add_argument("receiver", type=Path); p.add_argument("donor", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path); p.add_argument("--donor-state", type=Path)
    p.add_argument("--cue", choices=("A", "B", "C"), default="A"); p.add_argument("--dry", action="store_true")
    p.add_argument("--student-only", action="store_true"); p.add_argument("--alpha-only", action="store_true")
    a=p.parse_args()
    if a.mode == "teacher":
        if a.state or a.donor_state or a.cue != "A" or a.dry or a.student_only: p.error("Invalid teacher control arguments")
        teacher(a.receiver, a.donor, a.output, alpha_only=a.alpha_only)
    else:
        if a.alpha_only: p.error("Use --alpha-only with teacher mode")
        probe(a.receiver, a.donor, a.output, state=a.state, donor_state=a.donor_state,
              cue=a.cue, well=not a.dry, student_only=a.student_only)
