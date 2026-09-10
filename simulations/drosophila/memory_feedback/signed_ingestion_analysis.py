"""Audit the signed-domain course separately from its compatibility check."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from . import ingestion_analysis as analysis, ingestion_course as course
from . import signed_terminal_credit as signed
from ..connectome import sha256


def equal_arrays(left, right):
    a, b = np.load(left, mmap_mode="r"), np.load(right, mmap_mode="r")
    return a.shape == b.shape and all(np.array_equal(a[t:t+256], b[t:t+256]) for t in range(0, len(a), 256))


def run(base, source, output):
    metadata = course.read(source / "manifest.json")
    variant = metadata["signed_variant"]
    assert variant["source_sha256"] == sha256(Path(signed.__file__))
    assert variant["conversion"]["all_other_state_and_rng_exact"]
    with zipfile.ZipFile(source / "initial/state.paula") as archive:
        checkpoint = json.loads(archive.read("manifest.json"))
    assert checkpoint["sources"][str(Path(signed.__file__).resolve())] == variant["source_sha256"]
    result = analysis.run(base, source, output)
    result["signed_variant"] = variant
    result["signed_analysis_sha256"] = sha256(Path(__file__))
    compatibility = base / "memory-ingestion-signed-compatibility-20260910"
    checked = analysis.checked_record(compatibility / "check")
    q = np.load(compatibility / "check/release.npy", mmap_mode="r")
    result["compatibility_check_only"] = dict(manifest_sha256=sha256(compatibility / "manifest.json"),
        summary_sha256=sha256(compatibility / "check/summary.json"), completed_ticks=200,
        minimum_terminal=float(q.min()), ingested_j=checked["ingested_j"],
        limit="Separate near-boundary checkpoint migration, not the source of the scientific acquisition")
    original = base / "memory-ingestion-comparison-20260910/acquisition"
    result["fresh_acquisition_matches_original_positive_domain"] = {
        path.name: equal_arrays(path, source / "acquisition" / path.name) for path in original.glob("*.npy")}
    assert all(result["fresh_acquisition_matches_original_positive_domain"].values())
    result["signed_terminal_occupancy"] = {}
    for path in sorted(source.glob("*/release.npy")):
        q = np.load(path, mmap_mode="r")
        result["signed_terminal_occupancy"][path.parent.name] = dict(
            minimum=float(q.min()), final_negative_count=int(np.count_nonzero(q[-1] < 0)))
    result["signed_interpretation"] = "Credit contracts the magnitude of native signed information. Native positive-arrival filtering remains, so negative single-source release is ineffective, not biological inhibition. The variant does not establish restoration of silent terminals or a transmitter-level reconstruction."
    if (source / "memory-availability.json").exists():
        availability = course.read(source / "memory-availability.json")
        result["original_A_memory_availability"] = dict(receipt_sha256=sha256(source / "memory-availability.json"),
            interpretation=availability["interpretation"], conditions={})
        for label, condition in availability["conditions"].items():
            checked = analysis.checked_record(source / ("availability-"+label))
            assert condition["all_other_state_and_rng_exact"]
            assert condition["summary_sha256"] == sha256(source / ("availability-"+label) / "summary.json")
            result["original_A_memory_availability"]["conditions"][label] = dict(
                ingested_j=checked["ingested_j"], stored_energy_gain_j=checked["stored_energy_gain_j"])
    if result.get("action"):
        result["A_return_contact_ticks"] = {}
        for label in ("omission-intact", "omission-cut"):
            body = np.load(source / ("probe-"+label+"-A") / "body.npy", mmap_mode="r")
            contacts = np.flatnonzero(body[:, 5] > 0)
            result["A_return_contact_ticks"][label] = int(contacts[0]) if len(contacts) else None
        result["continued_food_control_A_return"] = {
            label: result["action"]["food-"+label]["retained_food_probes"]["A"] for label in ("intact", "cut")}
        result["continued_food_control_recorded_states_exact_before_return_probe"] = {
            stage: all(equal_arrays(path, source / (stage+"-cut") / path.name)
                       for path in (source / (stage+"-intact")).glob("*.npy"))
            for stage in ("action-food", "retention-food")}
        assert all(result["continued_food_control_recorded_states_exact_before_return_probe"].values())
        if result["omission_energy_benefit_j"] > 0 and result["retained_A_food_change_j"] < 0:
            result["interpretation"] = "The learned omission comparison persists and its fixed inhibitory action route saves energy while food is absent. The same route delays subsequent A-guided food collection and fails the joint criterion. Original A memory remains functionally available when that route is cut, as shown by its same-state terminal substitution. The comparison works; blanket inhibition does not provide adequate continued use of the retained memory."
            result["next_decision"] = "Stop this unit-pathway contrast without retuning. The next organizational question is how to distinguish loss of ongoing food contact from the action needed to obtain contact again. A contact-rate mismatch alone does not decide whether movement should stop."
    if result.get("action"):
        plot(source, result, output)
        result["figure_sha256"] = sha256(output / "ingestion-comparison.png")
    course.write(output / "ingestion-comparison.json", result)
    return result


def plot(source, result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = dict(omission="#a8462a", food="#276c8e", erased="#747474", untrained="#759347")
    for name, color in colors.items():
        data = np.load(source / ("verify-"+name) / "comparison.npy", mmap_mode="r")
        axes[0, 0].plot(np.arange(len(data))*.004, data[:, 3], label=name, color=color, lw=1.4)
    axes[0, 0].axvspan(.6, 2.4, color="#eeeeee", zorder=-1)
    axes[0, 0].set(title="Learned comparison during continued cue", ylabel="Negative-channel release", xlabel="Model seconds after outcome change")
    axes[0, 0].legend(fontsize=8)
    for name, color in (("intact", "#a8462a"), ("cut", "#276c8e")):
        path = source / ("action-omission-"+name)
        body = np.load(path / "body.npy", mmap_mode="r")
        initial = sum(course.read(path / "summary.json")["initial_organs"][:2])
        time = np.arange(len(body))*.004
        axes[0, 1].plot(time, body[:, 0], color=color, label=name)
        axes[1, 0].plot(time, body[:, 7:9].sum(axis=1)-initial, color=color, label=name)
    axes[0, 1].set(title="Omission: physical action", ylabel="Hinge angle, rad", xlabel="Model seconds")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(title="Omission: retained energy", ylabel="Energy + gut change, J", xlabel="Model seconds")
    cues = ("A", "B", "C")
    for offset, name, color in ((-.18, "intact", "#a8462a"), (.18, "cut", "#276c8e")):
        values = [result["action"]["omission-"+name]["retained_food_probes"][cue]["ingested_j"] for cue in cues]
        axes[1, 1].bar(np.arange(3)+offset, values, .36, color=color, label=name)
    axes[1, 1].set(title="Food probes after omission and retention", ylabel="Ingested energy, J", xticks=np.arange(3), xticklabels=cues)
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Signed-information composition: one fixed omission/action contrast", fontsize=12)
    fig.savefig(output / "ingestion-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.base, args.source, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("action", "signed_terminal_occupancy")}, indent=2))
