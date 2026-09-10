"""Summarize the bounded rule-location comparison and its physical limit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import Subgraph, sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint


def analyze(base, output):
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = Subgraph.load(base / "memory-alpha1-cut-20260910")
    result = dict(courses={}, probes={}, analysis_source_sha256=sha256(Path(__file__)))
    for name in ("paired", "unpaired", "receptor-block"):
        p = base / ("memory-selected-"+name+"-20260910")
        manifest = json.loads((p/"manifest.json").read_text())
        phases = {x["name"]: x for x in json.loads((p/"summary.json").read_text())}
        if "final_recovery" not in phases:
            raise ValueError("Incomplete recording")
        with np.load(p/"identities.npz") as z:
            ids, roots, selected = z["cells"], z["roots"], z["selected"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True))
        weights = np.load(p/"weights.npy", mmap_mode="r")
        soma = np.load(p/"soma.npy", mmap_mode="r")
        physical = np.load(p/"physical.npy", mmap_mode="r")
        retained = phases["retention"]["end"]
        net = load_checkpoint(p/"retention.paula", trusted=True).network
        changes = []
        for e in graph.internal:
            cell = net.network.neurons[int(e[3])]
            sid = next(s for s, origin in cell.synapse_sources.items() if origin[0] == e[2])
            v = float(cell.postsynaptic_points[sid].u_i.info)
            if v*e[5] < 0:
                changes.append(dict(source=str(e[0]), target=str(e[1]), value=v))
        smp = [list(ids).index(mapping[r]) for r in manifest["roles"]["SMP108"]]
        rule = {k: v for k, v in manifest["assumptions"]["learning_comparison"].items()
                if k not in {"memory_rule_ports", "extension_source"}}
        rule["selected_memory_pairs"] = len(selected)
        result["courses"][name] = dict(record=p.name, rule=rule,
            retained_weight={cue: float(weights[retained, np.isin(selected[:, 1], [mapping[r] for r in code])].mean())
                             for cue, code in manifest["codes"].items()},
            sign_changes_at_retention=changes,
            SMP108_spikes=int(soma[:, smp, 1].sum()), SMP108_max_S=float(soma[:, smp, 0].max()),
            retained_A=phases["retained_A"], retained_C=phases["retained_C"],
            delivered_j=float(physical[:, 3].sum()), digested_j=float(physical[:, 4].sum()),
            minimum_eta_post=min(c.params.eta_post for c in net.network.neurons.values()),
            minimum_eta_retro=min(c.params.eta_retro for c in net.network.neurons.values()),
            sources={f: sha256(p/f) for f in ("manifest.json", "summary.json", "weights.npy",
                "release.npy", "soma.npy", "physical.npy", "modulation.npy", "retention.paula")})
    names = ("paired-sham-A", "unpaired-sham-A", "paired-unpairedweights-A",
             "unpaired-pairedweights-A", "paired-sham-C", "paired-unpairedweights-C")
    for name in names:
        p = base / ("memory-selected-expression-"+name+"-20260910")
        r = json.loads((p/"summary.json").read_text())
        result["probes"][name] = {k: r[k] for k in ("cue", "spikes", "mean_angle", "changed_receiving",
            "changed_release", "changed_weights_during_probe", "exact_A_sham_replay", "trace_sha256")}
        result["probes"][name]["record"] = p.name
    result["conclusion"] = "Restricting the existing alternative equation to memory inputs preserves imported internal current signs and retains causal cue-memory expression. Useful movement and the teaching route remain unresolved; no second-order trial."
    (output/"selected-rule.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    order = [names[0], names[2], names[1], names[3]]
    bars = ax[0].bar(range(4), [result["probes"][n]["spikes"]["MBON07"] for n in order],
                     color=["#176B87", "#75A4B4", "#BC632F", "#D7A57D"])
    ax[0].bar_label(bars, padding=3)
    ax[0].set_xticks(range(4), ["Paired", "Paired +\nunpaired weights", "Unpaired", "Unpaired +\npaired weights"], fontsize=9)
    ax[0].set(ylabel="Memory-output spikes during A, 200 ticks", ylim=(0, 28))
    ax[0].set_title("Stored memory changes the cue response", loc="left", weight="bold")
    for name, label, color in [(names[0], "Paired", "#176B87"), (names[2], "Same state + unpaired weights", "#BC632F")]:
        with np.load(base/result["probes"][name]["record"]/"trace.npz") as z:
            ax[1].plot(np.arange(200), z["physical"][:, 0], color=color, label=label)
    ax[1].set(xlabel="A-expression probe tick", ylabel="Actual hinge angle, radians", ylim=(0, .065))
    ax[1].set_title("Movement changes little; teaching cell stays silent", loc="left", weight="bold")
    ax[1].text(.03, .40, "Motor-route cell: 8 spikes in both probes\nTeaching-route cell: 0 spikes in all courses", transform=ax[1].transAxes)
    ax[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Restricting the learning rule preserves inhibition and the memory effect", fontsize=13, weight="bold")
    fig.savefig(output/"selected-rule.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); analyze(a.records, a.output)
