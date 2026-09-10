"""Compact evidence from completed recordings; no rerunning or fitting dynamics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import Subgraph, sha256
from ...active_inference.core.runtime_checkpoint import load_checkpoint


NAMES = ("native-paired", "native-unpaired", "hebb-paired", "hebb-unpaired", "hebb-receptor-block")


def analyze(base, output):
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = Subgraph.load(base / "memory-alpha1-cut-20260910")
    results = {}; arrays = {}; probes = {}
    for name in NAMES:
        directory = base / ("memory-"+name+"-spaced-20260910")
        manifest = json.loads((directory / "manifest.json").read_text())
        phases = json.loads((directory / "summary.json").read_text())
        if phases[-1]["name"] != "final_recovery":
            raise ValueError("Incomplete acquisition course")
        phase = {p["name"]: p for p in phases}
        with np.load(directory / "identities.npz", allow_pickle=False) as z:
            selected, ids, roots = z["selected"], z["cells"], z["roots"]
        root_to_id = dict(zip(roots.tolist(), ids.tolist(), strict=True))
        rows = {int(nid): i for i, nid in enumerate(ids)}
        weights = np.load(directory / "weights.npy", mmap_mode="r")
        soma = np.load(directory / "soma.npy", mmap_mode="r")
        physical = np.load(directory / "physical.npy", mmap_mode="r")
        modulation = np.load(directory / "modulation.npy", mmap_mode="r")
        masks = {cue: np.isin(selected[:, 1], [root_to_id[r] for r in code])
                 for cue, code in manifest["codes"].items()}
        retained = phase["retention"]["end"]
        retention_begin = phase["retention"]["begin"]
        net = load_checkpoint(directory / "retention.paula", trusted=True).network
        sign_changes = []
        for e in graph.internal:
            cell = net.network.neurons[int(e[3])]
            sid = next(s for s, origin in cell.synapse_sources.items() if origin[0] == e[2])
            weight = float(cell.postsynaptic_points[sid].u_i.info)
            if weight * e[5] < 0:
                sign_changes.append(dict(source=str(e[0]), target=str(e[1]),
                    source_type=graph.nodes[str(e[0])]["annotation"]["hemibrain_type"],
                    target_type=graph.nodes[str(e[1])]["annotation"]["hemibrain_type"],
                    source_row=int(e[8]), initial=float(e[6]*.02), retained=weight))
        smp = [rows[root_to_id[r]] for r in manifest["roles"]["SMP108"]]
        mb = [rows[root_to_id[r]] for r in manifest["roles"]["MBON07"]]
        results[name] = dict(record=directory.name,
            ticks=len(soma), retained_weight={cue: float(weights[retained, mask].mean()) for cue, mask in masks.items()},
            weight_change_during_retention=float(np.max(np.abs(weights[retained]-weights[retention_begin]))),
            test_responses={cue: phase["retained_"+cue] for cue in ("A", "C")},
            dopamine_at_retained_A=float(modulation[retained, mb, 1].max()),
            SMP108_total_spikes=int(soma[:, smp, 1].sum()), SMP108_max_S=float(soma[:, smp, 0].max()),
            sign_changes=sign_changes, delivered_j=float(physical[:, 3].sum()),
            digested_j=float(physical[:, 4].sum()), maximum_energy_debt=float(physical[:, 7].max()),
            minimum_eta_post=min(c.params.eta_post for c in net.network.neurons.values()),
            minimum_eta_retro=min(c.params.eta_retro for c in net.network.neurons.values()),
            records_sha256={f: sha256(directory/f) for f in
                ("manifest.json", "summary.json", "soma.npy", "modulation.npy", "weights.npy",
                 "release.npy", "physical.npy", "identities.npz", "retention.paula", "retention-body.npz")})
        # Only compact plotting arrays retained after each run.
        arrays[name] = dict(weight_A=np.asarray(weights[::10, masks["A"]].mean(axis=1)),
                           weight_C=np.asarray(weights[::10, masks["C"]].mean(axis=1)),
                           retention=retained, phase=phase)
    for name in ("paired-sham-A", "unpaired-sham-A", "paired-unpairedweights-A",
                 "unpaired-pairedweights-A", "paired-sham-C", "paired-unpairedweights-C"):
        path = base / ("memory-expression-"+name+"-20260910")
        report = json.loads((path / "summary.json").read_text())
        probes[name] = {k: report[k] for k in ("cue", "changed_receiving", "changed_release", "spikes",
                         "mean_angle", "changed_weights_during_probe", "exact_A_sham_replay", "trace_sha256")}
        probes[name]["record"] = path.name
    native_w = [np.load(base/results[n]["record"]/"weights.npy", mmap_mode="r") for n in NAMES[:2]]
    facts = dict(anatomy=graph.summary(), courses=results, expression=probes,
        native_pairing_weights_exactly_equal=bool(np.array_equal(*native_w)),
        conclusion="Pairing-dependent stored coefficient effects with an existing alternative PAULA rule, but no accepted useful movement or second-order learning. The alternative also reverses four imported inhibitory current signs.",
        limits=["One deterministic anatomical selection and cue order; no population inference",
                "Controlled KC currents, unfitted dopamine assignment and engineered hinge motor boundary",
                "Retention measured in model ticks; no biological duration claim",
                "Expression probes remain adaptive and their final responses include new within-probe learning",
                "Student compartment is outside the running cut; second-order hypothesis is untested"],
        analysis_source_sha256=sha256(Path(__file__)))
    (output / "first-acquisition.json").write_text(json.dumps(facts, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = {"paired": "#176B87", "unpaired": "#BC632F"}
    for i, prefix in enumerate(("native", "hebb")):
        for history in ("paired", "unpaired"):
            d = arrays[prefix+"-"+history]
            ax[0, i].plot(np.arange(len(d["weight_A"]))*10, d["weight_A"],
                          label=history, color=colors[history], lw=2, ls="-" if history=="paired" else "--")
        a = arrays[prefix+"-paired"]["phase"]["retention"]
        ax[0, i].axvspan(a["begin"], a["end"], color="#DDE8D6", alpha=.6, label="retention")
        ax[0, i].set(xlabel="Continuing model tick", ylabel="Mean A → memory input weight", ylim=(0, 1.05))
        ax[0, i].legend(frameon=False, fontsize=9, loc="upper left")
    ax[0, 0].set_title("Native rule: identical stored cue weights", loc="left", weight="bold")
    ax[0, 1].set_title("Existing alternative: a retained pairing effect", loc="left", weight="bold")
    labels = ["Paired", "Paired +\nunpaired weights", "Unpaired", "Unpaired +\npaired weights"]
    order = ["paired-sham-A", "paired-unpairedweights-A", "unpaired-sham-A", "unpaired-pairedweights-A"]
    counts = [probes[n]["spikes"]["MBON07"] for n in order]
    bars = ax[1, 0].bar(range(4), counts, color=[colors["paired"], "#75A4B4", colors["unpaired"], "#D7A57D"])
    ax[1, 0].bar_label(bars, padding=3)
    ax[1, 0].set_xticks(range(4), labels, fontsize=9)
    ax[1, 0].set(ylabel="Memory-output spikes during A, 200 ticks", ylim=(0, 155))
    ax[1, 0].set_title("Stored coefficients cause the response change", loc="left", weight="bold")
    for n, label, color in [(order[0], "Paired", colors["paired"]),
                             (order[1], "Same state + unpaired weights", colors["unpaired"])]:
        with np.load(base/probes[n]["record"]/"trace.npz") as z:
            ax[1, 1].plot(np.arange(200), z["physical"][:, 0], color=color, label=label)
    ax[1, 1].set(xlabel="A-expression probe tick", ylabel="Actual hinge angle, radians", ylim=(0, .065))
    ax[1, 1].set_title("Small motor effect; no useful action established", loc="left", weight="bold")
    ax[1, 1].legend(frameon=False, fontsize=9, loc="lower left")
    ax[1, 1].text(.03, .46, "Teaching-route cell SMP108:\n0 spikes in every complete course", transform=ax[1, 1].transAxes)
    fig.suptitle("A stored cue effect is not yet a working adaptive organism", fontsize=16, weight="bold")
    fig.savefig(output / "first-acquisition.png", dpi=170)
    plt.close(fig)
    return facts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); analyze(a.records, a.output)
