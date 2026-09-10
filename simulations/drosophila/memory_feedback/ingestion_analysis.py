"""Assess the bounded learned-comparison and coupled physical consequences."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import feeding, ingestion_course as course
from .second_order import ProjectionGate
from ..connectome import sha256


def checked_record(path):
    result = course.read(path / "summary.json")
    for name, digest in result["artifacts"].items():
        assert sha256(path / name) == digest
    assert result["minimum_eta_post"] > 0 and result["minimum_eta_retro"] > 0
    assert result["branch_rng_preserved"]
    return result


def physical(path, manifest):
    report = checked_record(path)
    soma = np.load(path / "soma.npy", mmap_mode="r")
    body = np.load(path / "body.npy", mmap_mode="r")
    index = {nid: row for row, nid in enumerate(report["cells"])}
    motor = [index[n] for n in manifest["motor_ids"]]
    organ_delta = np.asarray(report["final_organs"])-report["initial_organs"]
    return dict(ingested_j=report["ingested_j"], stored_energy_gain_j=report["stored_energy_gain_j"],
        energy_demand_j=float(organ_delta[3]), unmet_j=float(organ_delta[4]), spill_j=float(organ_delta[5]),
        positive_work_and_activation_cost_j=float(organ_delta[3]-.2*.004*len(body)),
        motor_output_sum=float(soma[:, motor, 1].sum()), maximum_angle=float(body[:, 0].max()),
        gate=report["gate"], summary_sha256=sha256(path / "summary.json"))


def run(base, source, output):
    metadata = course.read(source / "manifest.json")
    _, manifest, mapping, selected = course.settings(base)
    metadata["motor_ids"] = [mapping[r] for r in manifest["roles"]["SMP108"]]
    if not (source / "verification.json").exists():
        return failed_continuation(base, source, output, metadata, manifest, mapping, selected)
    verification = course.read(source / "verification.json")
    for label in verification["results"]:
        checked_record(source / ("verify-"+label))
    checked_record(source / "acquisition")
    result = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        manifest_sha256=sha256(source / "manifest.json"), verification_sha256=sha256(source / "verification.json"),
        learned_persistent_comparison=verification["learned_persistent_comparison"],
        late_negative_means={name: r["late_negative_means"] for name, r in verification["results"].items()},
        action={})
    mask = {cue: np.isin(selected[:, 1], [mapping[r] for r in manifest["codes"][cue]]) for cue in ("A", "B", "C")}
    original = np.load(base / "memory-coverage-eight-intact-20260910/release.npy", mmap_mode="r")[-1]
    learned = np.load(source / "acquisition/release.npy", mmap_mode="r")[-1]
    result["original_memory_after_comparison_acquisition"] = {
        cue: dict(maximum_terminal_change=float(np.abs(learned[v]-original[v]).max()),
                  mean_before=float(original[v].mean()), mean_after=float(learned[v].mean())) for cue, v in mask.items()}
    pq = np.load(source / "acquisition/prediction_weights.npy", mmap_mode="r")
    result["predictor_acquisition"] = dict(initial_maximum=float(pq[0].max()), final_maximum=float(pq[-1].max()),
        final_mean_by_cue={cue: float(pq[-1, np.isin(metadata["contexts"], [mapping[r] for r in manifest["codes"][cue]])].mean()) for cue in mask})
    if (source / "action.json").exists():
        result["action_start_sha256"] = sha256(source / "action-start/state.paula")
        result["action_wiring"] = course.read(source / "action-wiring.json")
        for label in ("omission-intact", "omission-cut", "food-intact", "food-cut"):
            item = physical(source / ("action-"+label), metadata)
            item["retained_food_probes"] = {cue: physical(source / ("probe-"+label+"-"+cue), metadata) for cue in mask}
            q = np.load(source / ("retention-"+label) / "release.npy", mmap_mode="r")[-1]
            item["original_memory_maximum_change_since_acquisition"] = {cue: float(np.abs(q[v]-learned[v]).max()) for cue, v in mask.items()}
            result["action"][label] = item
        a, c = (result["action"]["omission-"+name] for name in ("intact", "cut"))
        result["omission_energy_benefit_j"] = a["stored_energy_gain_j"]-c["stored_energy_gain_j"]
        result["retained_A_food_change_j"] = a["retained_food_probes"]["A"]["ingested_j"]-c["retained_food_probes"]["A"]["ingested_j"]
        result["retained_B_food_change_j"] = a["retained_food_probes"]["B"]["ingested_j"]-c["retained_food_probes"]["B"]["ingested_j"]
        result["retained_cue_energy_changes_j"] = {cue: a["retained_food_probes"][cue]["stored_energy_gain_j"]-c["retained_food_probes"][cue]["stored_energy_gain_j"] for cue in ("A", "B")}
        result["ongoing_food_energy_change_j"] = result["action"]["food-intact"]["stored_energy_gain_j"]-result["action"]["food-cut"]["stored_energy_gain_j"]
        result["bounded_joint_criterion"] = bool(result["omission_energy_benefit_j"] > 1e-12 and
            result["retained_A_food_change_j"] >= -1e-12 and result["retained_B_food_change_j"] >= -1e-12 and
            min(result["retained_cue_energy_changes_j"].values()) >= -1e-8 and result["ongoing_food_energy_change_j"] >= -1e-8 and
            result["action"]["food-intact"]["ingested_j"] >= result["action"]["food-cut"]["ingested_j"]-1e-12)
        result["interpretation"] = (
            "At this fixed bound, the learned comparison pathway saves energy during omission while preserving tested A/B food intake and ongoing food collection. This is a limited coupled benefit, not general revaluation or net-utility learning."
            if result["bounded_joint_criterion"] else
            "The learned comparison is present, but the fixed inhibitory route does not meet the joint physical criterion of omission energy benefit and retained A/B food collection. Reduced output alone is not successful adaptation. Interpret this bound before changing the mechanism.")
    else:
        result["interpretation"] = verification["next_decision"]
    result["limits"] = "One retained preparation, ongoing contact-rate expectation, one predeclared unit inhibitory action route. No future-outcome timing, subjective disappointment, A-only revision of B, anatomical reconstruction of the added cells, or energetic-cost prediction is established. The original accepted A-to-B reference is unchanged."
    output.mkdir(parents=True, exist_ok=True)
    course.write(output / "ingestion-comparison.json", result)
    return result


def failed_continuation(base, source, output, metadata, manifest, mapping, selected):
    """Reproduce the recorded boundary failure with comparison traffic cut."""
    from neuron.neuron import setup_neuron_logger, RetrogradeSignalEvent
    setup_neuron_logger("CRITICAL")
    acquisition = checked_record(source / "acquisition")
    inverse = {nid: root for root, nid in mapping.items()}
    results, traces = {}, {}
    for cut in (False, True):
        branch, organism = course.restore(source / "acquisition")
        topo = branch.network.network
        gate = ProjectionGate(branch.network, set(topo.neurons), set(metadata["ids"].values()), cut)
        completed = []
        for t in range(200):
            before = {(int(e[1]), int(e[2])): float(topo.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info) for e in selected}
            slot = branch.network.current_tick % branch.network.wheel_size
            incoming = [s.event for s in branch.network.retrograde_wheel[slot] if isinstance(s.event, RetrogradeSignalEvent)]
            try:
                body = course.composition.step(branch, organism, mapping, manifest, metadata, "A", well=True, gate=gate)
            except ValueError as exc:
                failures = []
                for cell in topo.neurons.values():
                    for group in getattr(cell, "terminal_credit_groups", ()):
                        for terminal in group["terminals"]:
                            value = float(cell.presynaptic_points[terminal].u_o.info)
                            if value < 0 or value > 100 or not np.isfinite(value):
                                events = [e for e in incoming if e.target_neuron_id == cell.id and e.target_terminal_id == terminal]
                                failures.append(dict(cell=int(cell.id), root=inverse[cell.id], group=group["name"],
                                    cue=next(cue for cue, roots in manifest["codes"].items() if inverse[cell.id] in roots),
                                    terminal=int(terminal), before_retrograde=before[cell.id, terminal], after_retrograde=value,
                                    eta_retro=cell.params.eta_retro,
                                    incoming=[dict(source=int(e.source_neuron_id), source_port=int(e.source_synapse_id),
                                                   error_info=float(e.error_vector[0])) for e in events]))
                assert str(exc) == "Invalid eligibility flow" and failures
                results["cut" if cut else "intact"] = dict(completed_ticks=t, failing_tick=int(branch.network.current_tick),
                    exception=str(exc), failing_terminals=failures, dropped_new_projection_events=gate.removed)
                break
            old_state = np.array([[getattr(topo.neurons[n], f) for f in feeding.SOMA_FIELDS] for n in metadata["original_neuron_ids"]])
            old_q = np.array([topo.neurons[int(e[1])].presynaptic_points[int(e[2])].u_o.info for e in selected])
            completed.append(np.concatenate((old_state.ravel(), old_q, body)))
        else:
            raise AssertionError("Recorded failure did not reproduce at the same bound")
        traces["cut" if cut else "intact"] = np.asarray(completed)
    assert np.array_equal(traces["cut"], traces["intact"])
    assert results["cut"]["failing_terminals"] == results["intact"]["failing_terminals"]
    assert results["cut"]["failing_tick"] == results["intact"]["failing_tick"]
    q = np.load(source / "acquisition/release.npy", mmap_mode="r")
    pq = np.load(source / "acquisition/prediction_weights.npy", mmap_mode="r")
    original = np.load(base / "memory-coverage-eight-intact-20260910/release.npy", mmap_mode="r")[-1]
    incomplete_body = np.load(source / "comparison-start/body.npy", mmap_mode="r")
    result = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        manifest_sha256=sha256(source / "manifest.json"), acquisition_summary_sha256=sha256(source / "acquisition/summary.json"),
        acquisition_checkpoint_sha256=sha256(source / "acquisition/state.paula"),
        completed_acquisition_ticks=acquisition["phases"][-1]["end"],
        acquisition_ingested_j=acquisition["ingested_j"], acquisition_stored_energy_gain_j=acquisition["stored_energy_gain_j"],
        retained_old_terminal_minimum=float(q[-1].min()),
        original_memory_change={cue: float(np.abs(q[-1, np.isin(selected[:, 1], [mapping[r] for r in roots])]-original[np.isin(selected[:, 1], [mapping[r] for r in roots])]).max()) for cue, roots in manifest["codes"].items()},
        predictor_mean_weights_by_cue={cue: float(pq[-1, np.isin(metadata["contexts"], [mapping[r] for r in roots])].mean()) for cue, roots in manifest["codes"].items()},
        incomplete_lead_in_completed_ticks=int(np.count_nonzero(incomplete_body[:, 7])),
        failure_replays=results, pre_failure_original_soma_terminals_and_body_exact_between_replays=True,
        learned_persistent_comparison=None, action_coupling_run=False,
        interpretation="Eight prescribed acquisition presentations completed. The shared A-food lead-in then failed after 11 completed ticks when native signed retrograde updates made three old alpha1 A terminals negative. TerminalCreditNeuron requires nonnegative release. The same failure and old-cell/body trajectory occur when every added comparison projection is forward-cut from the retained acquisition state. Predictor A weights increased, but persistent omission signaling and behavioral adaptation were not assessed. Stop at this bound; no clipping, gain change, extra exposure or action coupling was applied.",
        next_decision="Resolve the domain compatibility between native signed presynaptic return updates and the opt-in nonnegative terminal-credit composition in a separately declared version before extending this preparation's lifespan. Do not silently repair the accepted checkpoints or treat the incomplete lead-in as a negative behavioral result.",
        limits="The diagnostic cut is applied after acquisition and does not undo its history. The new graph has no forward projection into the original circuit during acquisition; returning events from its predictor address dedicated new KC terminals. This failure concerns continued operation beyond the accepted A-to-B bound and leaves that recorded result unchanged.",
        incomplete_artifact_hashes={f.name: sha256(f) for f in (source / "comparison-start").iterdir() if f.is_file()})
    output.mkdir(parents=True, exist_ok=True)
    course.write(output / "ingestion-comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.source, args.output), indent=2))
