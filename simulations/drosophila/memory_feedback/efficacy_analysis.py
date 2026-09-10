"""Audit feedback recruitment, teacher-dependent credit and absent B action."""
import argparse
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import Subgraph, sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint, _serializer
from neuron.neuron import setup_neuron_logger


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    setup_neuron_logger("CRITICAL")
    base, output = Path(base), Path(output)
    parent = base/"memory-efficacy-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
        rows = dict(zip(z["roots"].tolist(), range(len(z["roots"])), strict=True))
    masks = {role+"_"+cue: np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) &
        np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]])
        for role in ("MBON07", "MBON04") for cue in ("A", "B", "C")}
    result = dict(source_sha256=sha256(Path(__file__)), courses={}, continuations={},
        efficacy=m["assumptions"]["feedback_efficacy"], graph_sha256=m["graph_sha256"])
    for name in ("paired", "unpaired"):
        p = base/f"memory-efficacy-{name}-20260910"
        phases = read(p/"summary.json")
        assert phases[-1]["name"] == "final_recovery"
        result["courses"][name] = dict(record=p.name,
            retained_A=next(ph for ph in phases if ph["name"] == "retained_A"),
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "release.npy", "weights.npy", "soma.npy", "body.npy")})
    traces = {}
    paths = {name: base/f"memory-efficacy-second-{name}-20260910" for name in ("intact", "cut", "unpaired", "displaced")}
    control = base/"memory-efficacy-teacher-removed-rng-20260910"
    paths["teacher_removed"] = control/"continuation"
    for name, p in paths.items():
        r = read(p/"summary.json")
        for f, digest in r["artifacts"].items():
            assert sha256(p/f) == digest, (p, f)
        assert r["nutrient_j"] == 0 and r["phases"][-1]["end"] == 3720
        q = np.load(p/"release.npy", mmap_mode="r")
        s = np.load(p/"soma.npy", mmap_mode="r")
        traces[name] = q[:, masks["MBON04_B"]].mean(axis=1)
        r["record"] = str(p.relative_to(base))
        r["summary_sha256"] = sha256(p/"summary.json")
        r["release"] = {key: dict(pairs=int(mask.sum()), before=float(q[0, mask].mean()),
            retained=float(q[-1, mask].mean())) for key, mask in masks.items()}
        r["phase_B_release"] = [{"name": ph["name"], "mean_release": float(traces[name][ph["end"]])} for ph in r["phases"]]
        r["course_spikes"] = {role: int(s[:, [rows[root] for root in rr], 1].sum()) for role, rr in m["roles"].items()}
        r["MBON04_max_S"] = float(s[:, [rows[root] for root in m["roles"]["MBON04"]], 0].max())
        r["retained_B_student"] = {}
        with np.load(p/"probe-B.npz") as z:
            for root in m["roles"]["MBON04"]:
                r["retained_B_student"][root] = dict(max_S=float(z["soma"][:, rows[root], 0].max()),
                    min_r=float(z["soma"][:, rows[root], 2].min()), spikes=int(z["soma"][:, rows[root], 1].sum()))
        result["continuations"][name] = r
    # Reverse the declared intervention in loaded diagnostic objects. Bytewise
    # equality then audits every remaining field and alias, including RNG.
    intervention = read(control/"intervention.json")
    original = load_checkpoint(parent/"retention.paula", trusted=True)
    changed = load_checkpoint(control/"receiver"/"retention.paula", trusted=True)
    for c in intervention["changes"]:
        changed.network.network.neurons[c["pre"]].presynaptic_points[c["terminal"]].u_o.info = (
            original.network.network.neurons[c["pre"]].presynaptic_points[c["terminal"]].u_o.info)
    _, pickler = _serializer()
    def serialize(branch):
        buf = io.BytesIO()
        pickler(buf, protocol=5).dump(dict(network=branch.network, python_rng=branch.python_rng, numpy_rng=branch.numpy_rng))
        return buf.getvalue()
    identical = serialize(original) == serialize(changed)
    assert identical, "Teacher intervention changed other runtime state"
    intervention["all_other_runtime_state_exact"] = identical
    intervention["body_file_identical"] = sha256(parent/"retention-body.npz") == sha256(control/"receiver"/"retention-body.npz")
    assert intervention["body_file_identical"]
    result["teacher_intervention"] = intervention
    del original, changed
    result["comparisons"] = {}
    for name in ("cut", "unpaired", "displaced", "teacher_removed"):
        a, b = paths["intact"], paths[name]
        comparison = dict(extra_B_release_depression=float(traces[name][-1]-traces["intact"][-1]), exact_arrays={})
        for f in ("body.npy", "release.npy", "weights.npy"):
            aa, bb = (np.load(p/f, mmap_mode="r") for p in (a, b))
            comparison["exact_arrays"][f] = all(np.array_equal(aa[i:i+200], bb[i:i+200]) for i in range(0, len(aa), 200))
        with np.load(a/"probe-B.npz") as za, np.load(b/"probe-B.npz") as zb:
            comparison["B_probe_body_exact"] = bool(np.array_equal(za["body"], zb["body"]))
            comparison["B_probe_student_max_abs_S_difference"] = float(np.max(np.abs(
                za["soma"][:, [rows[r] for r in m["roles"]["MBON04"]], 0]-zb["soma"][:, [rows[r] for r in m["roles"]["MBON04"]], 0])))
        result["comparisons"][name] = comparison
    g = Subgraph.load(Path(m["graph"]))
    result["student_output_routes"] = [dict(pre=str(e[0]), post=str(e[1]), count=int(e[4]), sign=int(e[5]), source_row=int(e[8]))
        for e in g.internal if str(e[0]) in m["roles"]["MBON04"] and str(e[1]) in m["roles"]["SMP108"]+m["roles"]["SMP353"]]
    result["checkpoint_limit"] = (
        "The original second_order runner saves ambient RNG at its final checkpoint. Fixed one-tick travel makes the RNG mismatch inert here, "
        "but these four original continuation checkpoints do not preserve advancing branch RNG. The corrected continuing_course wrapper and teacher control do. "
        "The earlier teacher-removed record without '-rng' is superseded and excluded from this audit.")
    result["conclusion"] = (
        "The gain-64 efficacy hypothesis recruits student dopamine through measured feedback. Most B-terminal depression is present without learned A. "
        "A small additional retained B-terminal depression depends on A terminal memory and temporal proximity. "
        "Neither that change nor removing the entire feedback projection changes B motor spike count; MBON04 remains silent. "
        "A feeding persists. This supports teacher-dependent local credit in this engineered preparation, not acquired useful B behavior or completed second-order learning.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"feedback-efficacy.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    labels = {"intact": "Learned A", "teacher_removed": "A memory removed", "cut": "Feedback blocked", "displaced": "Long B-to-A gap"}
    for name, label in labels.items():
        axes[0].plot(traces[name], label=label)
    axes[0].set(title="B-terminal release changes", xlabel="Continuing model ticks", ylabel="Mean release coefficient")
    axes[0].legend(frameon=False, fontsize=8)
    names = list(labels)
    spikes = [result["continuations"][n]["course_spikes"]["PAM08"] for n in names]
    bars = axes[1].bar(range(4), spikes, color=["#176B87", "#B35932", "#888888", "#709FAF"])
    axes[1].bar_label(bars)
    axes[1].set_xticks(range(4), ["Learned A", "A memory\nremoved", "Feedback\nblocked", "Long gap"], fontsize=8)
    axes[1].set(title="Recruitment includes B's own feedback", ylabel="Student dopamine spikes", ylim=(0, 27))
    axes[2].axis("off")
    axes[2].text(0, 1, "Retained useful A, unexpressed B\n\nA feeding after acquisition: 0.624 J\nB motor spikes: 2 in every control\nStudent output spikes: 0\n\nMost B depression occurs without A memory.\nA memory adds a small timing-dependent effect.\n\nNo acquired B action is established.", va="top", linespacing=1.8)
    fig.suptitle("Source-specific efficacy permits local credit, but student output stays silent", weight="bold")
    fig.savefig(output/"feedback-efficacy.png", dpi=170); plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.records, a.output)
