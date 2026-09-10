"""Audit broader primary teaching and the remaining cue/readout limitations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import Subgraph, sha256


def read(path):
    return json.loads(Path(path).read_text())


def run(base, output):
    base, output = Path(base), Path(output)
    parent = base/"memory-primary-paired-20260910"
    m = read(parent/"manifest.json")
    with np.load(parent/"identities.npz") as z:
        selected = z["selected"]
        mapping = dict(zip(z["roots"].tolist(), z["cells"].tolist(), strict=True))
        rows = dict(zip(z["roots"].tolist(), range(len(z["roots"])), strict=True))
    masks = {role+"_"+cue: np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]]) &
        np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]])
        for role in ("MBON07", "MBON04") for cue in ("A", "B", "C")}
    result = dict(source_sha256=sha256(Path(__file__)), primary_boundary=m["assumptions"]["primary_boundary"],
        graph_sha256=m["graph_sha256"], courses={}, continuation={}, expression_bound={})
    for name in ("paired", "unpaired"):
        p = base/f"memory-primary-{name}-20260910"
        phases = read(p/"summary.json")
        assert phases[-1]["end"] == 10120
        retention = next(ph["end"] for ph in phases if ph["name"] == "retention")
        q = np.load(p/"release.npy", mmap_mode="r")
        result["courses"][name] = dict(record=p.name,
            retained_A=next(ph for ph in phases if ph["name"] == "retained_A"),
            retained_release={k: float(q[retention, mask].mean()) for k, mask in masks.items()},
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "release.npy", "weights.npy", "soma.npy", "body.npy")})
    paths = {name: base/f"memory-primary-second-{name}-20260910" for name in ("intact", "cut", "unpaired", "displaced")}
    for name, p in paths.items():
        r = read(p/"summary.json")
        assert r["branch_rng_preserved"] and r["nutrient_j"] == 0 and r["phases"][-1]["end"] == 3720
        assert r["primary_boundary_sha256"] == m["assumptions"]["primary_boundary"]["source_sha256"]
        for f, digest in r["artifacts"].items():
            assert sha256(p/f) == digest
        q = np.load(p/"release.npy", mmap_mode="r")
        r["record"] = p.name; r["summary_sha256"] = sha256(p/"summary.json")
        r["release"] = {k: dict(before=float(q[0, mask].mean()), retained=float(q[-1, mask].mean())) for k, mask in masks.items()}
        r["course_spikes"] = {role: sum(ph["spikes"][role] for ph in r["phases"]) for role in m["roles"]}
        result["continuation"][name] = r
    result["comparisons"] = {}
    for name in ("cut", "unpaired", "displaced"):
        comparison = dict(extra_mean_B_depression=result["continuation"][name]["release"]["MBON04_B"]["retained"]-
            result["continuation"]["intact"]["release"]["MBON04_B"]["retained"])
        with np.load(paths["intact"]/"probe-B.npz") as a, np.load(paths[name]/"probe-B.npz") as b:
            comparison["B_body_exact"] = bool(np.array_equal(a["body"], b["body"]))
        result["comparisons"][name] = comparison
    for name in ("sham", "zero"):
        p = base/f"memory-student-bound-{name}-20260910"
        r = read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        result["expression_bound"][name] = r
    with np.load(base/"memory-student-bound-sham-20260910"/"trace.npz") as a, np.load(base/"memory-student-bound-zero-20260910"/"trace.npz") as b:
        result["expression_bound"]["body_exact"] = bool(np.array_equal(a["body"], b["body"]))
    result["expression_bound"]["limit"] = "Separate earlier student-output parent, independent 200-tick dry probes. Maximum terminal depression changes SMP108 but not this hinge trajectory; not a bound on every future state or behavior."
    result["student_memory_expression"] = {}
    for name in ("sham", "cutrelease"):
        p = base/f"memory-primary-expression-{name}-20260910"
        r = read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        result["student_memory_expression"][name] = r
    with np.load(base/"memory-primary-expression-sham-20260910"/"trace.npz") as a, np.load(
            base/"memory-primary-expression-cutrelease-20260910"/"trace.npz") as b, np.load(paths["intact"]/"probe-B.npz") as recorded:
        assert np.array_equal(a["soma"], recorded["soma"]) and np.array_equal(a["body"], recorded["body"])
        assert np.array_equal(a["weights"][0], b["weights"][0])
        assert np.array_equal(a["release"][0, ~masks["MBON04_B"]], b["release"][0, ~masks["MBON04_B"]])
        assert np.array_equal(b["release"][0, masks["MBON04_B"]], np.load(paths["cut"]/"release.npy", mmap_mode="r")[-1, masks["MBON04_B"]])
        result["student_memory_expression"]["comparison"] = dict(sham_exact=True,
            only_B_student_release_intervened=True, body_exact=bool(np.array_equal(a["body"], b["body"])),
            max_abs_difference_by_field={field: float(np.abs(a["soma"][:, :, i]-b["soma"][:, :, i]).max())
                for i, field in enumerate(m["soma_fields"])},
            different_spike_bins_by_role={role: int(np.count_nonzero(a["soma"][:, [rows[r] for r in m["roles"][role]], 1] !=
                b["soma"][:, [rows[r] for r in m["roles"][role]], 1])) for role in ("MBON04", "SMP108", "SMP353", "PAM07", "PAM08")})
    graph = Subgraph.load(Path(m["graph"]))
    result["original_direct_cue_routes"] = {cue: [dict(pre=str(e[0]), post=str(e[1]), count=int(e[4]), source_row=int(e[8]))
        for e in graph.internal if str(e[0]) in roots and str(e[1]) in m["roles"]["SMP108"]] for cue, roots in m["codes"].items()}
    balanced_path = base/"memory-balanced-codes-20260910"
    balanced = Subgraph.load(balanced_path)
    assert graph.selected == balanced.selected and graph.nodes == balanced.nodes and np.array_equal(graph.edges, balanced.edges)
    result["cue_reassignment"] = dict(manifest_sha256=sha256(balanced_path/"manifest.json"),
        exact_anatomical_files={f: sha256(Path(m["graph"])/f) == sha256(balanced_path/f) for f in ("nodes.json", "edges.npz")},
        specification=balanced.provenance["code_reassignment"],
        status="Prospective course; no learning outcome included in this audit")
    assert all(result["cue_reassignment"]["exact_anatomical_files"].values())
    result["conclusion"] = (
        "Broader primary teaching depresses A's student terminals and restores A-period neural feedback with active student outputs and retained A feeding. "
        "B-terminal changes remain dominated by B's own feedback; no acquired B movement appears. "
        "An independent maximum-depression diagnostic exposes the hinge readout's insensitivity in the tested state, and anatomy reveals a cue-assignment asymmetry. "
        "The next controlled cue panel balances declared anatomical loads without deleting any connection or fitting behavior.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"primary-boundary.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    names = ("intact", "cut", "unpaired", "displaced")
    food = [result["continuation"][n]["probes"]["A"]["well_j"] for n in names]
    bars=ax[0].bar(range(4), food, color=["#176B87", "#709FAF", "#B35932", "#888888"])
    ax[0].bar_label(bars, fmt="%.3f")
    ax[0].set_xticks(range(4), ["Intact", "Feedback\nblocked", "Unpaired", "Long gap"], fontsize=8)
    ax[0].set(title="A feeding survives continuing learning", ylabel="A-probe ingested energy, J", ylim=(0, .8))
    for n, label in (("intact", "Learned A"), ("unpaired", "Unpaired control")):
        q=np.load(paths[n]/"release.npy", mmap_mode="r")
        ax[1].plot(q[:, masks["MBON04_B"]].mean(axis=1), label=label)
    ax[1].set(title="B's own feedback dominates local change", xlabel="Continuing model ticks", ylabel="Mean B terminal release")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].axis("off")
    ax[2].text(0, 1, "Two remaining assay limitations\n\nMaximum student B depression:\nSMP108 spikes 16 → 17\nHinge trajectory exactly unchanged\n\nOriginal direct KC → SMP108 contacts:\nA: 0   B: 13   C: 0\n\nNew controlled codes: 4, 5, 4 contacts\nAll anatomical files remain identical.", va="top", linespacing=1.8)
    fig.suptitle("Primary teaching restores the teacher, but useful B behavior remains absent", weight="bold")
    fig.savefig(output/"primary-boundary.png", dpi=170); plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.records, a.output)
