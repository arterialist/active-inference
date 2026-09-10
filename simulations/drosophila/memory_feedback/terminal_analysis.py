"""Audit stored terminal credit, continued action, and unresolved recruitment."""
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
    graph = Subgraph.load(base/"memory-terminal-cut-20260910")
    result = dict(graph=graph.summary(), graph_sha256=sha256(base/"memory-terminal-cut-20260910"/"manifest.json"),
        source_sha256=sha256(Path(__file__)), courses={}, expression={}, continuation={})
    for name in ("paired", "unpaired", "disabled"):
        p = base/f"memory-terminal-{name}-20260910"; m = read(p/"manifest.json"); phases = read(p/"summary.json")
        assert phases[-1]["name"] == "final_recovery"
        retained = next(x["end"] for x in phases if x["name"] == "retention")
        with np.load(p/"identities.npz") as z:
            selected, ids, roots = z["selected"], z["cells"], z["roots"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True)); rows = dict(zip(roots.tolist(), range(len(roots))))
        q = np.load(p/"release.npy", mmap_mode="r"); w = np.load(p/"weights.npy", mmap_mode="r"); ss = np.load(p/"soma.npy", mmap_mode="r")
        coeff = {}
        for role in ("MBON07", "MBON04"):
            for cue in ("A", "B", "C"):
                mask = np.isin(selected[:, 3], [mapping[r] for r in m["roles"][role]]) & np.isin(selected[:, 1], [mapping[r] for r in m["codes"][cue]])
                coeff[role+"_"+cue] = dict(pairs=int(mask.sum()), initial_release=float(q[0, mask].mean()), retained_release=float(q[retained, mask].mean()),
                    initial_receiving=float(w[0, mask].mean()), retained_receiving=float(w[retained, mask].mean()))
        result["courses"][name] = dict(record=p.name, complete_ticks=phases[-1]["end"], retention_tick=retained,
            coefficients=coeff, assumptions={k: v for k, v in m["assumptions"]["terminal_credit"].items() if k != "groups"},
            pre_food_spikes={role: int(ss[:retained, [rows[r] for r in rr], 1].sum()) for role, rr in m["roles"].items()},
            retained_A=next(x for x in phases if x["name"] == "retained_A"),
            artifacts={f: sha256(p/f) for f in ("manifest.json", "summary.json", "identities.npz", "retention.paula", "soma.npy", "release.npy", "weights.npy", "body.npy")})
        if name == "paired":
            result["receptor_groups"] = m["assumptions"]["terminal_credit"]["groups"]
    for name in ("sham-A", "unpairedrelease-A", "pairedrelease-A", "sham-C", "dry-A", "dry-B"):
        p = base/f"memory-terminal-expression-{name}-20260910"; r = read(p/"summary.json")
        assert sha256(p/"trace.npz") == r["trace_sha256"]
        result["expression"][name] = r
    for name in ("intact", "cut", "unpaired"):
        p = base/f"memory-terminal-second-{name}-20260910"; r = read(p/"summary.json")
        for f, digest in r["artifacts"].items(): assert sha256(p/f) == digest
        assert r["nutrient_j"] == 0 and r["phases"][-1]["name"] == "retention"
        m = read(base/r["receiver"]/"manifest.json")
        with np.load(p/"identities.npz") as z:
            selected, ids, roots = z["selected"], z["cells"], z["roots"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True)); rows = dict(zip(roots.tolist(), range(len(roots))))
        ss = np.load(p/"soma.npy", mmap_mode="r"); q = np.load(p/"release.npy", mmap_mode="r")
        r["course_spikes"] = {role: int(ss[:, [rows[root] for root in rr], 1].sum()) for role, rr in m["roles"].items()}
        r["release_change"] = {}
        for role, cue in (("MBON07", "A"), ("MBON04", "B")):
            mask = np.isin(selected[:, 3], [mapping[x] for x in m["roles"][role]]) & np.isin(selected[:, 1], [mapping[x] for x in m["codes"][cue]])
            r["release_change"][role+"_"+cue] = dict(start=float(q[0, mask].mean()), end=float(q[-1, mask].mean()),
                changed_pairs=int(np.any(q[1:, mask] != q[0, mask], axis=0).sum()))
        result["continuation"][name] = r
    intact = base/"memory-terminal-second-intact-20260910"; cut = base/"memory-terminal-second-cut-20260910"
    result["intact_cut_equality"] = {}
    for f in ("weights.npy", "release.npy", "body.npy"):
        a, b = (np.load(p/f, mmap_mode="r") for p in (intact, cut))
        result["intact_cut_equality"][f] = bool(all(np.array_equal(a[i:i+200], b[i:i+200]) for i in range(0, len(a), 200)))
    result["projection_probes"] = {}
    for name in ("paired", "unpaired"):
        p = base/f"memory-terminal-projection-{name}-20260910"; r = read(p/"summary.json")
        for key, value in r["probes"].items(): assert sha256(p/(key+".npz")) == value["trace_sha256"]
        result["projection_probes"][name] = r
    # Diagnostic linearization only, before a DAN spike changes the network:
    # intact-minus-cut isolates the observed projection contribution. This
    # predicts a FIRST threshold crossing under fixed other inputs, not a
    # validated coupled-network response or a biologically measured efficacy.
    p = base/"memory-terminal-projection-paired-20260910"
    with np.load(p/"A-intact.npz") as z: a = z["soma"]
    with np.load(p/"A-cut.npz") as z: b = z["soma"]
    m = read(base/"memory-terminal-paired-20260910"/"manifest.json")
    with np.load(base/"memory-terminal-paired-20260910"/"identities.npz") as z:
        rows = dict(zip(z["roots"].tolist(), range(len(z["roots"]))))
    targets = {str(e[1]) for e in graph.internal if str(e[0]) in m["roles"]["SMP108"] and
               str(e[1]) in m["roles"]["PAM07"]+m["roles"]["PAM08"]}
    predictions = []
    for root in sorted(targets):
        row = rows[root]
        assert not np.any(a[:, row, 1]) and not np.any(b[:, row, 1])
        delta = a[:, row, 0]-b[:, row, 0]
        mask = delta > 1e-9
        if np.any(mask):
            values = np.where(mask, np.divide(a[:, row, 2]-b[:, row, 0], delta, out=np.full(200, np.inf), where=mask), np.inf)
            tick = int(np.argmin(values))
            predictions.append(dict(root=root, probe_tick=tick, critical_gain=float(values[tick]),
                background_S=float(b[tick, row, 0]), projection_delta_S=float(delta[tick])))
    result["projection_linearization"] = dict(targets=predictions,
        smallest_critical_gain=min(x["critical_gain"] for x in predictions),
        limit="Held-input first-crossing estimate only. Any changed efficacy must be declared and causally tested; contact counts remain anatomical evidence, not efficacy measurements.")
    result["conclusion"] = (
        "A new opt-in terminal-credit preparation stores a release-dependent direct memory and retains its useful action through the continuing B-before-A challenge. "
        "The comparison changes the learning locus, restores native receiving rules/rates, and includes measured DAN-to-KC providers; it is not an isolated attribution to trace duration. "
        "B is not acquired: student DANs remain silent, and removing 960 forward feedback events leaves memory coefficients and movement unchanged. "
        "The next unresolved stage is converting the memory-dependent feedback into student dopamine recruitment and an expressible student memory, not recovering A after extinction.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"terminal-credit.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for name, label, color in (("sham-A", "Learned A", "#176B87"), ("unpairedrelease-A", "Replace terminal memory", "#B35932"), ("sham-C", "Control C", "#777777")):
        with np.load(base/f"memory-terminal-expression-{name}-20260910"/"trace.npz") as z:
            ax[0].plot(np.arange(200)*.004, z["body"][:, 0], label=label, color=color)
    ax[0].axhline(.04, color="black", ls=":", lw=1)
    ax[0].set(title="Terminal memory causes feeding", xlabel="Expression time, seconds", ylabel="Hinge angle, radians")
    ax[0].legend(frameon=False, fontsize=8)
    vals = [result["expression"]["sham-A"]["well_j"], result["continuation"]["intact"]["probes"]["A"]["well_j"],
            result["continuation"]["cut"]["probes"]["A"]["well_j"], result["continuation"]["unpaired"]["probes"]["A"]["well_j"]]
    bars = ax[1].bar(range(4), vals, color=["#176B87", "#176B87", "#709FAF", "#B35932"])
    ax[1].bar_label(bars, fmt="%.3f", padding=3)
    ax[1].set_xticks(range(4), ["Before B", "After B", "After B,\nfeedback cut", "Unpaired\ncontrol"], fontsize=8)
    ax[1].set(title="A action survives continuing exposure", ylabel="A-probe food ingested, joules", ylim=(0, .75))
    ax[2].axis("off")
    ax[2].text(0, 1, "Remaining transfer failure\n\n960 feedback events removed\n0 student dopamine spikes\nIdentical selected memory coefficients\nand body trajectories with the cut\n\nB motor spikes: 2 before, 2 after\nA feeding: retained at 0.624 J\n\nThe next stage is student recruitment.\nNo claim of B learning.", va="top", linespacing=1.7)
    fig.suptitle("Local terminal credit preserves useful memory while adaptation continues", weight="bold")
    fig.savefig(output/"terminal-credit.png", dpi=170); plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a=p.parse_args(); run(a.records,a.output)
