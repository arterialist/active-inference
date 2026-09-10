"""One bounded ingestion-comparison acquisition and conditional action contrast."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import feeding, ingestion_comparison as composition
from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, save_checkpoint
from neuron.neuron import setup_neuron_logger


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def settings(base):
    parent = base / "memory-coverage-eight-intact-20260910"
    manifest = read(base / "memory-coverage-paired-20260910/manifest.json")
    with np.load(parent / "identities.npz") as z:
        roots = z["roots"].tolist()
        mapping = dict(zip(roots, z["cells"].tolist(), strict=True))
        selected = z["selected"]
    return parent, manifest, mapping, selected


def restore(path):
    branch = load_checkpoint(path / "state.paula", trusted=True)
    organism = feeding.FeedingBody()
    organism.restore(path / "body-state.npz")
    return branch, organism


def snapshot(branch, organism, path):
    path.mkdir(parents=True, exist_ok=True)
    save_checkpoint(branch, path / "state.paula", sources=[composition.__file__])
    organism.save(path / "body-state.npz")


def record(branch, organism, manifest, mapping, selected, metadata, phases, output, *, gate=None):
    output.mkdir(parents=True, exist_ok=False)
    net = branch.network
    ids = list(net.network.neurons)
    cells = [net.network.neurons[n] for n in ids]
    predictor = net.network.neurons[metadata["ids"]["prediction"]]
    total = sum(t for _, _, _, t in phases)
    def array(name, shape):
        return np.lib.format.open_memmap(output / (name+".npy"), mode="w+", dtype=np.float64, shape=shape)
    arrays = dict(
        soma=array("soma", (total, len(ids), len(feeding.SOMA_FIELDS))),
        body=array("body", (total, len(feeding.BODY_FIELDS))),
        release=array("release", (total+1, len(selected))),
        weights=array("weights", (total+1, len(selected))),
        prediction_weights=array("prediction_weights", (total+1, len(metadata["contexts"]))),
        context=array("context", (total, len(metadata["contexts"]))),
        arrivals=array("arrivals", (total, len(metadata["contexts"]))),
        comparison=array("comparison", (total, 8)),
    )
    def coefficients(t):
        arrays["release"][t] = [net.network.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected]
        arrays["weights"][t] = [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])].u_i.info for e in selected]
        arrays["prediction_weights"][t] = [predictor.postsynaptic_points[s].u_i.info for s in predictor.prediction_ports]
    coefficients(0)
    initial_energy = float(organism.organs.energy_j+organism.organs.gut_j)
    initial_organs = organism.organs.state().tolist()
    reports = []
    t = 0
    for name, cue, well, ticks in phases:
        start = t
        for _ in range(ticks):
            arrays["body"][t] = composition.step(branch, organism, mapping, manifest, metadata, cue, well=well, gate=gate)
            arrays["soma"][t] = [[getattr(cell, f) for f in feeding.SOMA_FIELDS] for cell in cells]
            coefficients(t+1)
            arrays["context"][t] = predictor.prediction_context
            arrays["arrivals"][t] = predictor.prediction_arrivals
            arrays["comparison"][t] = [net.network.neurons[metadata["ids"][r]].O for r in ("observation", "prediction", "positive", "negative")]+[
                predictor.prediction_error, predictor.prediction_error_used, predictor.prediction_error_arrival, predictor.prediction_eta]
            if not all(np.isfinite(a[t]).all() for a in arrays.values()):
                raise FloatingPointError("Nonfinite ingestion comparison")
            t += 1
        report = dict(name=name, cue=cue, well=well, begin=start, end=t,
            ingested_j=float(arrays["body"][start:t, 4:6].sum()),
            negative_mean=float(arrays["comparison"][start:t, 3].mean()))
        reports.append(report)
        print(json.dumps(report), flush=True)
    for a in arrays.values():
        a.flush()
    snapshot(branch, organism, output)
    summary = dict(source_sha256=sha256(Path(__file__)), composition_sha256=sha256(Path(composition.__file__)),
        cells=ids, soma_fields=list(feeding.SOMA_FIELDS), body_fields=list(feeding.BODY_FIELDS),
        comparison_fields=["observed", "predicted", "positive", "negative", "error", "error_used", "error_arrival", "eta"],
        phases=reports, initial_organs=initial_organs, final_organs=organism.organs.state().tolist(),
        stored_energy_gain_j=float(organism.organs.energy_j+organism.organs.gut_j-initial_energy),
        ingested_j=float(arrays["body"][:, 4:6].sum()),
        minimum_eta_post=min(c.params.eta_post for c in cells), minimum_eta_retro=min(c.params.eta_retro for c in cells),
        branch_rng_preserved=True, gate=None if gate is None else dict(cut=gate.enabled, observed=gate.observed, removed=gate.removed),
        artifacts={f.name: sha256(f) for f in output.iterdir() if f.is_file()})
    assert summary["minimum_eta_post"] > 0 and summary["minimum_eta_retro"] > 0
    write(output / "summary.json", summary)
    return summary


def acquire(base, output):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(base).free < 4*1024**3+600*1024**2:
        raise RuntimeError("Shared-volume reserve")
    parent, manifest, mapping, selected = settings(base)
    branch = load_checkpoint(parent / "retention.paula", trusted=True)
    organism = feeding.FeedingBody()
    organism.restore(parent / "retention-body.npz")
    metadata = composition.append_comparison(branch, mapping, manifest)
    output.mkdir(parents=True)
    metadata.update(parent_checkpoint_sha256=sha256(parent / "retention.paula"),
        parent_body_sha256=sha256(parent / "retention-body.npz"),
        composition_sha256=sha256(Path(composition.__file__)), source_sha256=sha256(Path(__file__)),
        manifest=manifest,
        acquisition_bound="Eight A-with-well 200-tick presentations, each followed by 1000 blank ticks; 1000 final blank ticks. Then one shared 200-tick A-with-well lead-in before omission branches.",
        verification_bound="Independent 600-tick A-food, A-omission, A-omission with predictor context weights removed, and C-omission probes. Late windows 150:300 and 300:600 exclude onset/offset propagation transients.",
        action_bound="Only if learned comparison passes: one 2x2 contrast from the same shared learned state, omission versus food, intact versus forward-cut negative-to-SMP108 pathway, for 600 ticks. Then blank retention and independent cue probes; no gain or exposure sweep.")
    write(output / "manifest.json", metadata)
    snapshot(branch, organism, output / "initial")
    phases = [(f"pair{i}_{label}", cue, well, ticks) for i in range(8)
              for label, cue, well, ticks in (("A_food", "A", True, 200), ("blank", "", False, 1000))]
    phases.append(("retention", "", False, 1000))
    record(branch, organism, manifest, mapping, selected, metadata, phases, output / "acquisition")
    record(branch, organism, manifest, mapping, selected, metadata, [("A_food_lead_in", "A", True, 200)], output / "comparison-start")


def verify(base, output):
    _, manifest, mapping, selected = settings(base)
    metadata = read(output / "manifest.json")
    assert metadata["composition_sha256"] == sha256(Path(composition.__file__))
    source = output / "comparison-start"
    results = {}
    for name, cue, well, erased in (("food", "A", True, False), ("omission", "A", False, False),
                                   ("erased", "A", False, True), ("untrained", "C", False, False)):
        branch, organism = restore(source)
        changes = []
        if erased:
            predictor = branch.network.network.neurons[metadata["ids"]["prediction"]]
            before = composition.serialized(branch)
            points = [predictor.postsynaptic_points[s] for s in predictor.prediction_ports]
            originals = [p.u_i.info for p in points]
            for point, value in zip(points, originals):
                point.u_i.info = type(value)(0.)
            snapshot(branch, organism, output / "erased-start")
            for point, value in zip(points, originals):
                point.u_i.info = value
            assert composition.serialized(branch) == before
            changes = [float(x) for x in originals]
            branch, organism = restore(output / "erased-start")
        path = output / ("verify-"+name)
        results[name] = record(branch, organism, manifest, mapping, selected, metadata,
            [(name, cue, well, 600)], path)
        results[name]["predictor_weights_before_removal"] = changes
        comparison = np.load(path / "comparison.npy", mmap_mode="r")
        results[name]["late_negative_means"] = [float(comparison[a:b, 3].mean()) for a, b in ((150, 300), (300, 600))]
    negatives = {name: r["late_negative_means"] for name, r in results.items()}
    passed = all(negatives["omission"][i] > max(1e-6, 5*max(negatives[c][i] for c in ("food", "erased", "untrained"))) for i in (0, 1))
    result = dict(results=results, learned_persistent_comparison=passed,
        criterion="Omission negative-channel mean exceeds 1e-6 and five times each food, predictor-erased and untrained-cue control, separately over ticks 150:300 and 300:600. This is a signal identification bound, not predictor calibration or biological time inference.",
        next_decision="Proceed to the predeclared action contrast" if passed else "Stop at this acquisition bound and interpret the missing learned comparison before changing construction.")
    write(output / "verification.json", result)
    print(json.dumps(dict(verification=passed, late_negative_means=negatives)), flush=True)


def action(base, output):
    if not read(output / "verification.json")["learned_persistent_comparison"]:
        raise ValueError("Action coupling requires the learned-comparison contrast")
    _, manifest, mapping, selected = settings(base)
    metadata = read(output / "manifest.json")
    branch, organism = restore(output / "comparison-start")
    targets = [mapping[r] for r in manifest["roles"]["SMP108"]]
    edges = composition.append_action(branch, metadata, targets)
    snapshot(branch, organism, output / "action-start")
    write(output / "action-wiring.json", dict(edges=edges, role=metadata["action_role"],
        scope="Added receiving port expands each SMP108 input count and learning-window upper bound. Original experimental cue ports keep their old indices. Both contrast arms have exactly this same constructed state."))
    result = {}
    for outcome, well in (("omission", False), ("food", True)):
        for cut in (False, True):
            label = outcome+("-cut" if cut else "-intact")
            branch, organism = restore(output / "action-start")
            gate = composition.action_gate(branch, metadata, targets, cut)
            result[label] = record(branch, organism, manifest, mapping, selected, metadata,
                [(outcome, "A", well, 600)], output / ("action-"+label), gate=gate)
            retention = output / ("retention-"+label)
            record(branch, organism, manifest, mapping, selected, metadata, [("retention", "", False, 1000)], retention, gate=gate)
            result[label]["probes"] = {}
            for cue in ("A", "B", "C"):
                b, body = restore(retention)
                g = composition.action_gate(b, metadata, targets, cut)
                result[label]["probes"][cue] = record(b, body, manifest, mapping, selected, metadata,
                    [(cue+"_food", cue, True, 200)], output / ("probe-"+label+"-"+cue), gate=g)
    write(output / "action.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "verify", "action"))
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_neuron_logger("CRITICAL")
    dict(acquire=acquire, verify=verify, action=action)[args.mode](args.base, args.output)
