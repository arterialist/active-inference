"""Assess the one body-context comparison and its original-record screen."""
import argparse
from pathlib import Path

import numpy as np

from . import ingestion_course as course, ingestion_analysis as previous
from .feeding import FeedingBody, WELL_ANGLE
from ..connectome import sha256


def contact_summary(source, label):
    path = source / ("probe-"+label+"-A")
    body = np.load(path / "body.npy", mmap_mode="r")
    comparison = np.load(path / "comparison.npy", mmap_mode="r")
    initial = FeedingBody()
    initial.restore(source / ("retention-"+label) / "body-state.npz")
    position = np.r_[initial.body.data.qpos[0], body[:-1, 0]]
    before_contact = position < WELL_ANGLE
    hits = np.flatnonzero(body[:, 5] > 0)
    return dict(first_contact_tick=int(hits[0]) if len(hits) else None,
        precontact_ticks=int(before_contact.sum()), maximum_pre_step_angle=float(position.max()),
        precontact_negative_mean=float(comparison[before_contact, 3].mean()) if before_contact.any() else None,
        body_sha256=sha256(path / "body.npy"), comparison_sha256=sha256(path / "comparison.npy"))


def run(base, source, output):
    metadata = course.read(source / "manifest.json")
    _, manifest, mapping, selected = course.settings(base)
    metadata["motor_ids"] = [mapping[r] for r in manifest["roles"]["SMP108"]]
    acquisition = previous.checked_record(source / "acquisition")
    action = course.read(source / "action.json")
    assert len(action["action_edges"]) == 1 and action["action_edges"][0]["birth_weight"] == -1.
    reference = base / "memory-ingestion-signed-20260910"
    result = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        manifest_sha256=sha256(source / "manifest.json"), action_sha256=sha256(source / "action.json"),
        action_start_sha256=sha256(source / "action-start/state.paula"),
        body_context=metadata["body_context"], signed_variant=metadata["signed_variant"],
        original_record_screen={label: contact_summary(reference, label) for label in ("omission-intact", "omission-cut", "food-intact", "food-cut")},
        conditions={}, acquisition=dict(ingested_j=acquisition["ingested_j"], stored_energy_gain_j=acquisition["stored_energy_gain_j"]))
    result["original_record_screen"]["genuine_loss_minimum_angle"] = float(np.load(reference / "action-omission-intact/body.npy", mmap_mode="r")[:, 0].min())
    for label, original_report in action["conditions"].items():
        intervention = original_report["body_context_intervention"]
        assert intervention["all_other_state_and_rng_exact"]
        assert intervention["changed_readouts"] == (len(metadata["body_context"]["gates"]) if label.endswith("bypass") else 0)
        item = previous.physical(source / ("action-"+label), metadata)
        item["intervention"] = intervention
        item["retained_food_probes"] = {cue: previous.physical(source / ("probe-"+label+"-"+cue), metadata) for cue in ("A", "B")}
        item["A_return"] = contact_summary(source, label)
        comparison = np.load(source / ("action-"+label) / "comparison.npy", mmap_mode="r")
        item["late_negative_means"] = [float(comparison[a:b, 3].mean()) for a, b in ((150, 300), (300, 600))]
        result["conditions"][label] = item
    result["A_return_benefits"] = {}
    for outcome in ("omission", "food"):
        a, b = [result["conditions"][outcome+"-"+label]["retained_food_probes"]["A"] for label in ("body", "bypass")]
        result["A_return_benefits"][outcome] = dict(food_j=a["ingested_j"]-b["ingested_j"],
            retained_energy_j=a["stored_energy_gain_j"]-b["stored_energy_gain_j"])
    a, b = [result["conditions"]["omission-"+label] for label in ("body", "bypass")]
    result["established_loss_comparison_exact_between_body_and_bypass"] = bool(np.array_equal(
        np.load(source / "action-omission-body/comparison.npy", mmap_mode="r"),
        np.load(source / "action-omission-bypass/comparison.npy", mmap_mode="r")))
    body_loss_signal = all(a["late_negative_means"][i] > max(1e-6, .9*b["late_negative_means"][i]) for i in (0, 1))
    food_signal_quiet = max(result["conditions"]["food-body"]["late_negative_means"]) < 1e-6
    reacquisition = all(result["conditions"][outcome+"-body"]["retained_food_probes"]["A"]["stored_energy_gain_j"] > 0 and
        result["A_return_benefits"][outcome]["food_j"] > 0 and result["A_return_benefits"][outcome]["retained_energy_j"] > 0 for outcome in ("omission", "food"))
    b_preserved = all(result["conditions"][outcome+"-body"]["retained_food_probes"]["B"]["ingested_j"] >=
        result["conditions"][outcome+"-bypass"]["retained_food_probes"]["B"]["ingested_j"]-1e-12 for outcome in ("omission", "food"))
    result["criterion"] = dict(loss_signal_preserved=body_loss_signal, continued_food_signal_quiet=food_signal_quiet,
        useful_A_reacquisition_improves_in_both_histories=reacquisition, B_collection_preserved=b_preserved,
        bounded_body_context_benefit=body_loss_signal and food_signal_quiet and reacquisition and b_preserved,
        definition="Preserve at least 90% of the matched bypass's sustained loss signal in each existing late window, with continued-food signal below 1e-6. Require positive A-probe retained-energy gain and improved A food/energy versus bypass after both histories; B intake must not decline. This is a bounded comparison, not full restoration to every earlier reference performance.")
    initial = np.load(source / "acquisition/release.npy", mmap_mode="r")[0]
    q = np.load(source / "acquisition/release.npy", mmap_mode="r")[-1]
    result["old_memory_change_during_acquisition"] = {cue: float(np.abs(q[mask]-initial[mask]).max())
        for cue, mask in ((cue, np.isin(selected[:, 1], [mapping[r] for r in roots])) for cue, roots in manifest["codes"].items())}
    result["interpretation"] = (
        "At this fixed bound, actual joint-position context preserves the response to established food loss and improves useful A-food reacquisition while preserving B collection. The matched readout bypass establishes a causal role for body dependence."
        if result["criterion"]["bounded_body_context_benefit"] else
        "This fixed cue-by-position composition does not meet the joint loss-response and useful-reacquisition criterion. Inspect the physical and signal differences before changing architecture or tuning.")
    result["limits"] = "Existing nonlinear coincidence features with learned predictor weights, one physical afferent and one fixed well geometry. No learned sharp contact threshold, new brake gain, host approach/contact flag, grace timer, general reward timing or reconstructed anatomy. The prior omission-brake causal contrast supplies the route evidence; this comparison isolates body dependence at its coincidence readout."
    output.mkdir(parents=True, exist_ok=True)
    plot(source, result, output)
    result["figure_sha256"] = sha256(output / "body-context.png")
    course.write(output / "body-context.json", result)
    return result


def plot(source, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for label, color, style in (("body", "#276c8e", "-"), ("bypass", "#a8462a", "--")):
        path = source / ("action-omission-"+label)
        comparison = np.load(path / "comparison.npy", mmap_mode="r")
        axes[0, 0].plot(np.arange(len(comparison))*.004, comparison[:, 3], color=color, ls=style, label=label)
        for row, col, outcome in ((0, 1, "omission"), (1, 0, "food")):
            body = np.load(source / ("probe-"+outcome+"-"+label+"-A") / "body.npy", mmap_mode="r")
            axes[row, col].plot(np.arange(len(body))*.004, body[:, 0], color=color, ls=style, label=label)
    axes[0, 0].set(title="Established food contact removed", xlabel="Model seconds", ylabel="Negative comparison release")
    axes[0, 1].set(title="A return after omission", xlabel="Model seconds", ylabel="Hinge angle, rad")
    axes[1, 0].set(title="A return after continued-food control", xlabel="Model seconds", ylabel="Hinge angle, rad")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axhline(WELL_ANGLE, color="#888888", lw=1., ls=":", label="Collection angle")
    groups = (("omission", "A"), ("food", "A"), ("omission", "B"), ("food", "B"))
    for offset, label, color in ((-.18, "body", "#276c8e"), (.18, "bypass", "#a8462a")):
        values = [result["conditions"][outcome+"-"+label]["retained_food_probes"][cue]["stored_energy_gain_j"] for outcome, cue in groups]
        axes[1, 1].bar(np.arange(4)+offset, values, .36, color=color, label=label)
    axes[1, 1].set(title="Retained energy in return probes", ylabel="Energy + gut gain, J",
        xticks=np.arange(4), xticklabels=["A / loss", "A / food", "B / loss", "B / food"])
    axes[1, 1].axhline(0, color="#888888", lw=.8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("Actual joint-position context versus matched coincidence-readout bypass", fontsize=12)
    fig.savefig(output / "body-context.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import json
    print(json.dumps(run(args.base, args.source, args.output), indent=2))
