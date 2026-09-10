"""Audit the balanced cue panel and calibrate one eligibility-window test."""
import argparse
import io
import json
from pathlib import Path

import numpy as np

from ..connectome import sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    setup_neuron_logger("CRITICAL")
    base, output = Path(base), Path(output)
    parent = base/"memory-balanced-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
        rows = dict(zip(z["roots"].tolist(), range(len(z["roots"])), strict=True))
    masks = {role+"_"+cue: np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) &
        np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]])
        for role in ("MBON07", "MBON04") for cue in ("A", "B", "C")}
    result = dict(source_sha256=sha256(Path(__file__)), graph_sha256=m["graph_sha256"], courses={}, continuation={})
    for name in ("paired", "unpaired"):
        p = base/f"memory-balanced-{name}-20260910"
        phases = read(p/"summary.json")
        assert phases[-1]["end"] == 10120
        retention = next(ph["end"] for ph in phases if ph["name"] == "retention")
        q = np.load(p/"release.npy", mmap_mode="r")
        result["courses"][name] = dict(record=p.name,
            retained_A=next(ph for ph in phases if ph["name"] == "retained_A"),
            retained_release={k: float(q[retention, mask].mean()) for k, mask in masks.items()},
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "release.npy", "weights.npy", "soma.npy", "body.npy")})
    for name in ("intact", "cut", "unpaired", "displaced"):
        p = base/f"memory-balanced-second-{name}-20260910"
        r = read(p/"summary.json")
        assert r["branch_rng_preserved"] and r["nutrient_j"] == 0 and r["phases"][-1]["end"] == 3720
        for f, digest in r["artifacts"].items():
            assert sha256(p/f) == digest
        q = np.load(p/"release.npy", mmap_mode="r")
        r["record"] = p.name; r["summary_sha256"] = sha256(p/"summary.json")
        r["release"] = {k: dict(before=float(q[0, mask].mean()), retained=float(q[-1, mask].mean())) for k, mask in masks.items()}
        r["course_spikes"] = {role: sum(ph["spikes"][role] for ph in r["phases"]) for role in m["roles"]}
        if name == "intact":
            s = np.load(p/"soma.npy", mmap_mode="r")
            r["first_A_event_ticks"] = {role: np.flatnonzero(np.any(s[160:360, [rows[root] for root in m["roles"][role]], 1] > 0, axis=1)).tolist()
                for role in ("SMP353", "SMP108", "PAM07", "PAM08")}
        result["continuation"][name] = r
    with np.load(base/"memory-balanced-second-intact-20260910"/"probe-B.npz") as a, np.load(base/"memory-balanced-second-cut-20260910"/"probe-B.npz") as b:
        result["intact_cut_B_body_exact"] = bool(np.array_equal(a["body"], b["body"]))
    result["expression_bound"] = {}
    for name in ("sham", "zero"):
        p = base/f"memory-balanced-bound-{name}-20260910"
        r = read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        with np.load(p/"trace.npz") as z:
            r["SMP108_max_S"] = float(z["soma"][:, [rows[root] for root in m["roles"]["SMP108"]], 0].max())
        result["expression_bound"][name] = r
    with np.load(base/"memory-balanced-bound-sham-20260910"/"trace.npz") as a, np.load(base/"memory-balanced-bound-zero-20260910"/"trace.npz") as b:
        result["expression_bound"]["body_exact"] = bool(np.array_equal(a["body"], b["body"]))
    # The next tau comparison must change only that parameter at birth.
    window = base/"memory-window-paired-20260910"
    wm = read(window/"manifest.json")
    original = load_checkpoint(parent/"birth.paula", trusted=True)
    changed = load_checkpoint(window/"birth.paula", trusted=True)
    count = 0
    for nid, cell in changed.network.network.neurons.items():
        if hasattr(cell, "terminal_credit_tau_kc"):
            assert cell.terminal_credit_tau_kc == 256.
            cell.terminal_credit_tau_kc = original.network.network.neurons[nid].terminal_credit_tau_kc
            count += 1
    _, pickler = _serializer()
    def blob(branch):
        buf = io.BytesIO()
        pickler(buf, protocol=5).dump(dict(network=branch.network, python_rng=branch.python_rng, numpy_rng=branch.numpy_rng))
        return buf.getvalue()
    exact = blob(original) == blob(changed)
    assert exact and count == 174
    result["prospective_window"] = dict(specification=wm["assumptions"]["eligibility_window"],
        configured_cells=count, all_other_birth_state_exact=exact,
        reference_birth_sha256=sha256(parent/"birth.paula"), changed_birth_sha256=sha256(window/"birth.paula"),
        completed_trace_fraction_after_171_ticks={"tau64": float(np.exp(-171/64)), "tau256": float(np.exp(-171/256))},
        status="Direct courses completed; nutrient-free continuing comparisons started. Their outcomes are not included here.")
    result["window_direct_courses"] = {}
    for name in ("paired", "unpaired"):
        p = base/f"memory-window-{name}-20260910"
        phases = read(p/"summary.json")
        assert phases[-1]["end"] == 10120
        retention = next(ph["end"] for ph in phases if ph["name"] == "retention")
        q = np.load(p/"release.npy", mmap_mode="r")
        result["window_direct_courses"][name] = dict(record=p.name,
            retained_A=next(ph for ph in phases if ph["name"] == "retained_A"),
            retained_release={k: float(q[retention, mask].mean()) for k, mask in masks.items()},
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "release.npy", "weights.npy", "soma.npy", "body.npy")})
    result["conclusion"] = (
        "Anatomy-based cue reassignment removes B's autonomous feedback in this panel while preserving retained A feeding. "
        "The four student dopamine spikes during acquisition now occur only during learned A, and disappear in the unpaired and projection-cut controls. "
        "B still does not acquire action. Even imposed zero gamma4 B release leaves SMP108 silent in the balanced dry probe, so the student-expression input/output boundaries also need reassessment. "
        "Recorded teacher arrival times motivate a separate longer-eligibility comparison, with direct acquisition rechecked because normalized EMA buildup also changes.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"balanced-codes.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.records, a.output)
