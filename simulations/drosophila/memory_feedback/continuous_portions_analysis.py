"""Audit continuous acquisition, physical renewal and its energetic cost."""
import argparse
from pathlib import Path
import zipfile
import json

import numpy as np

from . import continuous_portions as experiment, ingestion_course as course
from .ingestion_analysis import checked_record
from .ingestion_comparison import serialized
from ..connectome import sha256


def checkpoint_tick(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("manifest.json"))["tick"]


def verify_control(source, metadata):
    branch, body = experiment.restore(source / "start-retained")
    before = serialized((branch.network, branch.python_rng, branch.numpy_rng))
    del branch
    branch, _ = experiment.restore(source / "start-erased")
    predictor = branch.network.network.neurons[metadata["ids"]["prediction"]]
    assert all(predictor.postsynaptic_points[s].u_i.info == 0. for s in predictor.prediction_ports)
    values = course.read(source / "manifest.json")["control"]["original_prediction_weights"]
    for sid, value in zip(predictor.prediction_ports, values, strict=True):
        predictor.postsynaptic_points[sid].u_i.info = value
    assert serialized((branch.network, branch.python_rng, branch.numpy_rng)) == before
    with np.load(source / "start-retained/body-state.npz") as a, np.load(source / "start-erased/body-state.npz") as b:
        assert a.files == b.files
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    return dict(neural_state_and_rng_exact_except_stored_prediction_weights=True,
        physical_and_inventory_state_exact=True,
        start_checkpoint_sha256={label: sha256(source / ("start-"+label) / "state.paula") for label in ("retained", "erased")})


def analyze_arm(source, label, metadata, motor_ids):
    path = source / label
    summary = checked_record(path)
    summary_plus = course.read(source / (label+".json"))
    environment_path = source / (label+"-environment.npy")
    assert summary_plus["environment_sha256"] == sha256(environment_path)
    assert summary["phases"][0]["cue"] == "A" and summary["phases"][0]["well"]
    assert len(summary["phases"]) == 1
    body = np.load(path / "body.npy", mmap_mode="r")
    environment = np.load(environment_path, mmap_mode="r")
    comparison = np.load(path / "comparison.npy", mmap_mode="r")
    soma = np.load(path / "soma.npy", mmap_mode="r")
    weights = np.load(path / "prediction_weights.npy", mmap_mode="r")
    assert len(body) == len(environment) == experiment.TICKS
    assert np.isfinite(environment).all()
    initial = experiment.PortionBody()
    initial.restore(source / ("start-"+label) / "body-state.npz")
    final = experiment.PortionBody()
    final.restore(path / "body-state.npz")
    assert np.isclose(final.body.data.time-initial.body.data.time, len(body)*.004)
    assert checkpoint_tick(path / "state.paula")-checkpoint_tick(source / ("start-"+label) / "state.paula") == len(body)
    assert final.samples == len(body)
    assert np.array_equal(environment[:, 0], np.r_[initial.body.data.qpos[0], body[:-1, 0]])
    assert np.array_equal(environment[:, 2], body[:, 5]) and np.all(body[:, 4] == 0.)
    remaining, loaded = experiment.PORTION_J, 1
    for row in environment:
        angle, before, accepted, after, portions, refill, depleted = row
        should_refill = remaining == 0. and angle <= experiment.RELEASE_ANGLE
        assert bool(refill) == should_refill
        if should_refill:
            remaining, loaded = experiment.PORTION_J, loaded+1
        assert before == remaining and portions == loaded
        assert 0. <= accepted <= min(experiment.DOSE_J, remaining)
        assert accepted == 0. or angle >= experiment.WELL_ANGLE
        remaining = max(0., remaining-accepted)
        if remaining < 1e-12:
            remaining = 0.
        assert after == remaining and bool(depleted) == (before > 0. and remaining == 0.)
    assert final.remaining_j == remaining and final.portions_loaded == loaded
    intake = environment[:, 2] > 0.
    acquired = np.unique(environment[intake, 4]).astype(int).tolist()
    depletions = np.flatnonzero(environment[:, 6]).tolist()
    replenishments = np.flatnonzero(environment[:, 5]).tolist()
    motor_rows = [summary["cells"].index(n) for n in motor_ids]
    spikes = soma[:, motor_rows, 1].sum(axis=1)
    params = initial.organs.params
    positive_work = np.maximum(0., .2*body[:, 3]*np.diff(np.r_[initial.body.data.qpos[0], body[:, 0]]))
    activation = params.activation_w*body[:, 3]**2*.004
    work_cost = positive_work/params.efficiency
    demand = params.basal_w*.004+activation+work_cost
    assert np.isclose(demand.sum(), final.organs.demand_j-initial.organs.demand_j, rtol=0, atol=1e-8)
    energy = body[:, 7:9].sum(axis=1)
    complete_cycles = [dict(begin_replenishment_tick=a, end_replenishment_tick=b,
        seconds=(b-a)*.004, food_j=float(body[a:b, 5].sum()),
        retained_energy_gain_j=float(energy[b-1]-energy[a-1]),
        basal_j=float(params.basal_w*.004*(b-a)), activation_j=float(activation[a:b].sum()),
        positive_work_metabolic_j=float(work_cost[a:b].sum()))
        for a, b in zip(replenishments[:-1], replenishments[1:], strict=True)]
    start = depletions[0]+1 if depletions else len(body)
    post = slice(start, len(body))
    active = np.flatnonzero(spikes[post] > 0.)+start
    return dict(summary_sha256=sha256(path / "summary.json"), environment_sha256=sha256(environment_path),
        neural_and_body_elapsed_ticks=len(body), continuous_context="A, with no blanks or attempt resets",
        food_j=summary["ingested_j"], retained_energy_gain_j=summary["stored_energy_gain_j"],
        physical_energy_demand_j=float(final.organs.demand_j-initial.organs.demand_j),
        cost_breakdown_j=dict(basal=params.basal_w*.004*len(body), activation=float(activation.sum()),
            positive_work_metabolic=float(work_cost.sum())),
        unmet_energy_j=float(final.organs.unmet_j-initial.organs.unmet_j),
        spill_j=float(final.organs.spill_j-initial.organs.spill_j),
        complete_replenishment_to_replenishment_cycles=complete_cycles,
        acquired_portions=acquired, portions_loaded=loaded, remaining_j=remaining,
        first_ingestion_tick=int(np.flatnonzero(intake)[0]) if intake.any() else None,
        depletion_ticks=depletions, replenishment_ticks=replenishments,
        repeated_acquisition=len(acquired) >= 2,
        useful_repeated_acquisition=len(acquired) >= 2 and summary["stored_energy_gain_j"] > 0.,
        post_first_depletion=dict(ticks=len(body)-start,
            minimum_angle_rad=float(environment[post, 0].min()) if start < len(body) else None,
            mean_command=float(body[post, 3].mean()) if start < len(body) else None,
            motor_output_sum=float(spikes[post].sum()),
            maximum_gap_between_positive_output_or_window_edges=int(np.diff(np.r_[start, active, len(body)]).max()) if start < len(body) else None,
            mean_negative_comparison=float(comparison[post, 3].mean()) if start < len(body) else None),
        late_windows=[dict(begin=a, end=b, minimum_angle_rad=float(body[a:b, 0].min()),
            mean_command=float(body[a:b, 3].mean()), mean_negative_comparison=float(comparison[a:b, 3].mean()))
            for a, b in ((1000, 2000), (4000, 5000))],
        predictor_weights=dict(initial_mean=float(weights[0].mean()), final_mean=float(weights[-1].mean()),
            maximum_change=float(np.abs(weights[-1]-weights[0]).max())),
        terminal_memory_maximum_change=float(np.abs(np.load(path / "release.npy", mmap_mode="r")[-1]-np.load(path / "release.npy", mmap_mode="r")[0]).max()),
        minimum_eta_post=summary["minimum_eta_post"], minimum_eta_retro=summary["minimum_eta_retro"])


def plot(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for label, color in (("retained", "#276c8e"), ("erased", "#a8462a")):
        path = source / label
        body = np.load(path / "body.npy", mmap_mode="r")
        comparison = np.load(path / "comparison.npy", mmap_mode="r")
        summary = course.read(path / "summary.json")
        time = (np.arange(len(body))+1)*.004
        axes[0, 0].plot(time, body[:, 0], color=color, label=label)
        axes[0, 1].plot(time, np.cumsum(body[:, 5]), color=color, label=label)
        axes[1, 0].plot(time, comparison[:, 3], color=color, label=label)
        axes[1, 1].plot(time, body[:, 7:9].sum(axis=1)-sum(summary["initial_organs"][:2]), color=color, label=label)
    axes[0, 0].axhline(experiment.WELL_ANGLE, color="#555555", ls="--", lw=1., label="Collection")
    axes[0, 0].axhline(experiment.RELEASE_ANGLE, color="#777777", ls=":", lw=1., label="Replenishment")
    for ax, title, ylabel in zip(axes.flat,
        ("Physical engagement", "Food acquired", "Outcome comparison", "Continuing energetic return"),
        ("Hinge angle, rad", "Cumulative ingestion, J", "Negative comparison release", "Energy + gut gain, J"), strict=True):
        ax.set(title=title, xlabel="Continuous model time, s", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[1, 1].axhline(0, color="#777777", lw=.8)
    axes[0, 0].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Constant A, finite portions: stored expectation retained versus initially erased", fontsize=12)
    fig.savefig(output / "continuous-portions.png", dpi=160)
    plt.close(fig)


def run(base, source, output):
    _, manifest, mapping, _, metadata = experiment.settings(base)
    protocol = course.read(source / "manifest.json")
    assert protocol["source_sha256"] == sha256(Path(experiment.__file__))
    control = verify_control(source, metadata)
    results = {label: analyze_arm(source, label, metadata, [mapping[r] for r in manifest["roles"]["SMP108"]]) for label in ("retained", "erased")}
    retained, erased = results["retained"], results["erased"]
    result = dict(source_sha256=sha256(Path(__file__)), source=str(source.resolve()),
        manifest_sha256=sha256(source / "manifest.json"), protocol=protocol, control=control, conditions=results,
        stored_expectation_energy_benefit_j=retained["retained_energy_gain_j"]-erased["retained_energy_gain_j"],
        joint_function_passed=retained["useful_repeated_acquisition"] and
            len(retained["acquired_portions"]) > len(erased["acquired_portions"]),
        scope="One fixed continuous-context finite-portion environment and matched initial predictor-weight intervention. All adaptation active. No external timing of disengagement, cue phases or neural/body resets. Energy excludes a metabolic charge for the neural population.")
    if not any(v["repeated_acquisition"] for v in results.values()):
        result["interpretation"] = "Neither arm renews acquisition. Stored expectation changes the ongoing loss response, but useful self-organized feeding cycles are not established. Remaining above the physical replenishment boundary identifies a failure to terminate engagement within this bound, despite passive withdrawal being mechanically available."
    elif retained["repeated_acquisition"] and not erased["repeated_acquisition"] and not retained["useful_repeated_acquisition"]:
        result["interpretation"] = "The stored expectation causally supports repeated physical disengagement and renewed acquisition under constant sensory context. The erased arm persists beyond contact after its first portion. Retained cycles improve food and energy relative to that control, but total energy+gut still declines, so the joint useful-feeding criterion fails. This establishes expectation-dependent renewal in a fixed feedback circuit, not learning the action sequence itself; oscillation with learned coefficients remains a compatible mechanism."
    else:
        result["interpretation"] = "Assess repeated acquisition and net energy jointly. A stored-weight effect identifies a contribution of initial expectation; it does not by itself establish a learned action sequence or exclude fixed or newly learned feedback oscillation."
    output.mkdir(parents=True, exist_ok=True)
    plot(source, output)
    result["figure_sha256"] = sha256(output / "continuous-portions.png")
    course.write(output / "continuous-portions.json", result)
    print(json.dumps(dict(conditions=results, interpretation=result["interpretation"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.base, args.source, args.output)
