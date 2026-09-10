"""Audit completed feeding prerequisites and the bounded failed transfer test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import sha256


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(analysis_source_sha256=sha256(Path(__file__)), feeding={}, composition={}, transfer={})
    for name in ("sham-A", "unpairedweights-A", "pairedweights-A", "sham-C", "dry-A"):
        p = base/f"memory-feeding-expression-{name}-20260910"
        result["feeding"][name] = read(p/"summary.json")
        assert result["feeding"][name]["trace_sha256"] == sha256(p/"trace.npz")
    for name in ("paired", "unpaired"):
        p = base/f"memory-student-{name}-bounded-20260910"
        m = read(p/"manifest.json"); phases = read(p/"summary.json")
        assert phases[-1]["name"] == "final_recovery"
        with np.load(p/"identities.npz") as z:
            ids, roots, selected = z["cells"], z["roots"], z["selected"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
        rows = {int(n): i for i, n in enumerate(ids)}
        soma = np.load(p/"soma.npy", mmap_mode="r")
        weights = np.load(p/"weights.npy", mmap_mode="r")
        k = next(x["end"] for x in phases if x["name"] == "retention")
        roles = {}
        for role in ("MBON04", "PAM07", "PAM08"):
            ix = [rows[mapping[r]] for r in m["roles"][role]]
            roles[role] = dict(spikes=int(soma[:, ix, 1].sum()), max_S=float(soma[:, ix, 0].max()),
                              minimum_threshold=float(soma[:, ix, 2].min()))
        gamma = np.isin(selected[:, 3], [mapping[r] for r in m["roles"]["MBON04"]])
        result["composition"][name] = dict(record=p.name, roles=roles,
            student_memory_pairs=int(gamma.sum()), zero_student_weights_at_retention=int((weights[k, gamma] == 0).sum()),
            retained_A=next(x for x in phases if x["name"] == "retained_A"),
            sources={f: sha256(p/f) for f in ("manifest.json", "summary.json", "retention.paula", "soma.npy", "weights.npy")})
    gate = base/"memory-gate-expression-20260910"
    result["gate_expression"] = read(gate/"summary.json")
    for f, digest in result["gate_expression"]["traces"].items():
        assert sha256(gate/f) == digest
    for name in ("intact", "cut", "displaced", "no-A-memory"):
        p = base/f"memory-second-{name}-20260910"
        s = read(p/"summary.json")
        assert s["phases"][-1]["name"] == "retention" and s["nutrient_j"] == 0
        for f, digest in s["artifacts"].items():
            assert sha256(p/f) == digest, (name, f)
        parent = base/s["receiver"]; m = read(parent/"manifest.json")
        with np.load(p/"identities.npz") as z:
            ids, roots, selected = z["cells"], z["roots"], z["selected"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
        rows = {int(n): i for i, n in enumerate(ids)}
        soma = np.load(p/"soma.npy", mmap_mode="r"); mods = np.load(p/"modulation.npy", mmap_mode="r")
        weights = np.load(p/"weights.npy", mmap_mode="r")
        s["roles"] = {}
        for role in ("MBON04", "PAM07", "PAM08"):
            ix = [rows[mapping[r]] for r in m["roles"][role]]
            s["roles"][role] = dict(spikes=int(soma[:, ix, 1].sum()), max_S=float(soma[:, ix, 0].max()),
                minimum_threshold=float(soma[:, ix, 2].min()), maximum_abs_modulation=float(np.abs(mods[:, ix]).max()))
        s["memory_weights"] = {}
        for role in ("MBON07", "MBON04"):
            for cue in ("A", "B", "C"):
                mask = np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]]) & np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]])
                s["memory_weights"][role+"_"+cue] = dict(pairs=int(mask.sum()), start_mean=float(weights[0, mask].mean()),
                    end_mean=float(weights[-1, mask].mean()), zero_at_end=int((weights[-1, mask] == 0).sum()),
                    changed_pairs=int(np.any(weights[1:, mask] != weights[0, mask], axis=0).sum()))
                zeros = np.flatnonzero(np.all(weights[:, mask] == 0, axis=1))
                s["memory_weights"][role+"_"+cue]["first_all_zero_completed_ticks"] = int(zeros[0]) if len(zeros) else None
        result["transfer"][name] = s
    intact = base/"memory-second-intact-20260910"; cut = base/"memory-second-cut-20260910"
    equal = {}
    for f in ("body.npy", "weights.npy", "release.npy"):
        a, b = (np.load(d/f, mmap_mode="r") for d in (intact, cut))
        equal[f] = bool(all(np.array_equal(a[i:i+200], b[i:i+200]) for i in range(0, len(a), 200)))
    for cue in ("A", "B"):
        with np.load(intact/f"probe-{cue}.npz") as a, np.load(cut/f"probe-{cue}.npz") as b:
            equal["probe_"+cue] = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
    result["intact_cut_equality"] = equal
    result["conclusion"] = (
        "No acquired A-to-B teaching in this bounded preparation. The projection carries events but fails to recruit native student DANs; "
        "student MBONs never spike and their dopamine channel stays zero. Feedback interruption preserves original A feeding, "
        "and removing 374 acquisition events leaves all selected weight/release and physical trajectories unchanged. "
        "Continued cue exposure also eliminates A-guided movement in both branches. This is a composition failure, not evidence "
        "against second-order conditioning in flies or PAULA in general. No further tuning or circuit expansion.")
    (output/"bounded-transfer.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for name, label, color in (("sham-A", "Paired A", "#176B87"), ("unpairedweights-A", "Replace with unpaired coefficients", "#B35932"), ("sham-C", "Control cue C", "#808080")):
        with np.load(base/f"memory-feeding-expression-{name}-20260910"/"trace.npz") as z:
            ax[0, 0].plot(np.arange(200)*.004, z["body"][:, 0], label=label, color=color)
    ax[0, 0].axhline(.04, color="black", ls=":", lw=1, label="Food contact")
    ax[0, 0].set(title="First-order reconstruction is the prerequisite", xlabel="Probe time, seconds", ylabel="Hinge angle, radians")
    ax[0, 0].legend(frameon=False, fontsize=8)
    labels = ["Student output\nMBON04", "Student DAN\nPAM07", "Student DAN\nPAM08"]
    values = [result["composition"]["paired"]["roles"][k]["max_S"] for k in ("MBON04", "PAM07", "PAM08")]
    bars = ax[0, 1].bar(labels, values, color="#176B87")
    ax[0, 1].bar_label(bars, fmt="%.3f", padding=3)
    ax[0, 1].axhline(1, color="#B35932", ls="--", label="Native threshold")
    ax[0, 1].set(title="The added compartment never spikes", ylabel="Largest recorded somatic state", ylim=(0, 1.18))
    ax[0, 1].legend(frameon=False, fontsize=8)
    a = np.load(intact/"weights.npy", mmap_mode="r")
    m = read(base/result["transfer"]["intact"]["receiver"]/"manifest.json")
    with np.load(intact/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
    for key, label, color in (("MBON07_A", "Teacher inputs for A", "#176B87"), ("MBON04_B", "Student inputs for B", "#B35932")):
        role, cue = key.split("_")
        mask = np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]]) & np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]])
        ax[1, 0].plot(np.arange(len(a))*.004, a[:, mask].mean(axis=1), color=color, label=label)
    ax[1, 0].set(title="Intact and feedback-cut weights match exactly", xlabel="B-training course time, seconds", ylabel="Mean receiving coefficient")
    ax[1, 0].legend(frameon=False, fontsize=8)
    ax[1, 1].axis("off")
    g = result["gate_expression"]
    text = (f"Before B training\nA feeding survives the feedback cut: {g['branches'][0]['well_j']:.3f} J in both\n\n"
            "B-before-A acquisition\n0 J nutrients; 374 feedback events removed\n0 student DAN and MBON spikes in either branch\n\n"
            "After retention\nB: 0 motor spikes in both branches\nA: 0 motor spikes in both branches\n\n"
            "No demonstrated memory-mediated transfer.\nThe teaching route and continuing A expression fail.")
    ax[1, 1].text(0, 1, text, va="top", linespacing=1.55, fontsize=10)
    fig.suptitle("A functional first-order memory does not compose into a working teacher here", fontsize=13, weight="bold")
    fig.savefig(output/"bounded-transfer.png", dpi=170)
    plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); run(a.records, a.output)
