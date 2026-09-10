"""Separate student survival, recruitment, and feedback specificity."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome import Subgraph, sha256


def run(base, output):
    base, output = Path(base), Path(output)
    records = {"native": "memory-student-paired-bounded-20260910", **{
        n: f"memory-viability-{n}-20260910" for n in ("slow", "sensitive", "combined")}}
    result = dict(courses={}, probes={}, source_sha256=sha256(Path(__file__)))
    for name, record in records.items():
        p = base/record; m = json.loads((p/"manifest.json").read_text()); phases = json.loads((p/"summary.json").read_text())
        assert phases[-1]["name"] == "final_recovery"
        k = next(x["end"] for x in phases if x["name"] == "retention")
        with np.load(p/"identities.npz") as z:
            roots, ids, selected = z["roots"], z["cells"], z["selected"]
        mapping = dict(zip(roots.tolist(), ids.tolist(), strict=True)); rows = dict(zip(roots.tolist(), range(len(roots))))
        ss = np.load(p/"soma.npy", mmap_mode="r"); ww = np.load(p/"weights.npy", mmap_mode="r"); mm = np.load(p/"modulation.npy", mmap_mode="r")
        mask = np.isin(selected[:, 3], [mapping[r] for r in m["roles"]["MBON04"]])
        stats = {role: dict(spikes=int(ss[:k, [rows[r] for r in rr], 1].sum()),
            max_S=float(ss[:k, [rows[r] for r in rr], 0].max()),
            max_M1=float(mm[:k, [rows[r] for r in rr], 1].max())) for role, rr in m["roles"].items()}
        result["courses"][name] = dict(record=record, graph_sha256=m["graph_sha256"],
            parameters=m["assumptions"].get("student_viability", dict(student_eta_post=.01, DAN_sensitivity=1.)),
            complete_ticks=phases[-1]["end"], pre_food_ticks=k, pre_food_roles=stats,
            student_pairs=int(mask.sum()), zero_student_weights_at_retention=int((ww[k, mask] == 0).sum()),
            minimum_student_weight=float(ww[k, mask].min()),
            retained_A=next(x for x in phases if x["name"] == "retained_A"),
            source_hashes={f: sha256(p/f) for f in ("manifest.json", "summary.json", "retention.paula", "identities.npz", "soma.npy", "weights.npy", "modulation.npy", "body.npy")})
    assert len({r["graph_sha256"] for r in result["courses"].values()}) == 1
    for name in ("sensitive", "combined"):
        p = base/f"memory-viability-probe-{name}-20260910"
        r = json.loads((p/"summary.json").read_text())
        for key, v in r["probes"].items():
            assert v["nutrient_j"] == 0 and sha256(p/(key+".npz")) == v["trace_sha256"]
        result["probes"][name] = r
    result["interpretation"] = (
        "Slower receiving adaptation preserves all student inputs. Greater DAN sensitivity recruits dopamine but also responds strongly to novel B and control C, including with feedback cut. "
        "Together they yield six student MBON spikes before feeding and two in the retained A probe. These are viability gains, not memory-mediated transfer. "
        "The next decision concerns input-specific teaching efficacy and delayed local credit; a global DAN sensitivity change is insufficient.")
    result["mechanism_limit"] = dict(
        source="input_rule.py: MemoryInputRuleNeuron.tick, pending includes only positive current input_buffer arrivals",
        verified_test="tests/test_drosophila_student_viability.py::test_existing_selected_rule_has_no_delayed_credit_without_new_arrival",
        statement="After B input ends, a later local dopamine state cannot change that selected weight without another B arrival. Positive rates alone do not create delayed credit.",
        existing_extensions="EligibilityTraceNeuron requires postsynaptic spikes. PredictiveReceptorNeuron has contextual/error traces but requires graded release, two opponent zero-throughput error ports and bounded dynamics; it is not a drop-in replacement for this measured spiking MBON/DAN circuit.",
        biological_motivation="Hige et al. 2015 demonstrated dopamine-paired KC-to-MBON depression while suppressing MBON spikes in gamma1pedc. This motivates testing a local heterosynaptic mechanism, but does not identify gamma4 parameters or validate the present composition.",
        source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/")
    graph = Subgraph.load(base/"memory-alpha1-gamma4-cut-20260910")
    gamma = set(sum(graph.provenance["student_codes"].values(), []))
    mbons = set(graph.provenance["student_MBONS"])
    to_kc, to_mbon, feedback = Counter(), Counter(), Counter()
    for e in graph.edges:
        a, b, count = str(e[0]), str(e[1]), int(e[4])
        if a == graph.provenance["teaching_interneuron"]:
            feedback[b] += count
        if graph.nodes[a]["annotation"]["hemibrain_type"] in {"PAM07", "PAM08"}:
            if b in gamma: to_kc[a] += count
            if b in mbons: to_mbon[a] += count
    providers = [dict(root=r, type=graph.nodes[r]["annotation"]["hemibrain_type"],
        side=graph.nodes[r]["annotation"]["side"], in_cut=r in graph.selected,
        SMP108_contacts=feedback[r], to_gamma_KC_contacts=to_kc[r], to_MBON04_contacts=to_mbon[r])
        for r in sorted(set(to_kc)|set(to_mbon))]
    result["receptor_location_audit"] = dict(providers=providers,
        total_providers=len(providers), included_providers=sum(x["in_cut"] for x in providers),
        gamma_KC_contacts=sum(to_kc.values()), included_gamma_KC_contacts=sum(v for r, v in to_kc.items() if r in graph.selected),
        SMP108_contacts=sum(x["SMP108_contacts"] for x in providers),
        included_SMP108_contacts=sum(x["SMP108_contacts"] for x in providers if x["in_cut"]),
        inference="Selecting DANs by direct MBON04 afferents omits measured DAN-to-KC routes. These boundary records motivate reviewing receptor location before increasing arbitrary gain; they do not prove all omitted cells are needed.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"student-viability.json").write_text(json.dumps(result, indent=2)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
    order = list(records); labels = ["Native", "Slower\nlearning", "DAN\nsensitivity", "Both"]
    bars = axes[0].bar(range(4), [96-result["courses"][n]["zero_student_weights_at_retention"] for n in order], color="#176B87")
    axes[0].bar_label(bars); axes[0].set_xticks(range(4), labels, fontsize=9)
    axes[0].set(title="Slower adaptation prevents collapse", ylabel="Positive student inputs at retention", ylim=(0, 108))
    bars = axes[1].bar(range(4), [result["courses"][n]["pre_food_roles"]["MBON04"]["spikes"] for n in order], color="#B35932")
    axes[1].bar_label(bars); axes[1].set_xticks(range(4), labels, fontsize=9)
    axes[1].set(title="Both changes permit sparse output", ylabel="Student MBON spikes before feeding", ylim=(0, 8))
    rr = result["probes"]["combined"]["probes"]
    for offset, suffix, label, color in ((-.18, "intact", "Intact feedback", "#176B87"), (.18, "cut", "Feedback cut", "#B35932")):
        vals = [sum(rr[c+"-"+suffix]["roles"][r]["spikes"] for r in ("PAM07", "PAM08")) for c in ("A", "B", "C")]
        bars = axes[2].bar(np.arange(3)+offset, vals, width=.36, color=color, label=label)
        axes[2].bar_label(bars, fontsize=8, padding=2)
    axes[2].set_xticks(range(3), ["Learned A", "Novel B", "Control C"], fontsize=9)
    axes[2].set(title="Most dopamine activity survives the cut", ylabel="Student DAN spikes, dry 200-tick probe", ylim=(0, 260))
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Existing parameters repair student viability, but not a selective teacher", weight="bold")
    fig.savefig(output/"student-viability.png", dpi=170); plt.close(fig)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); run(a.records, a.output)
