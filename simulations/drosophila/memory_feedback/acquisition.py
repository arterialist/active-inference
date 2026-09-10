"""Continuing alpha1 cue/nutrient acquisition, with explicit engineered boundaries.

Uses existing PAULA equations only. A and C are controlled disjoint KC currents,
not reconstructed odors. Physical nutrient delivery drives the identified PAM11
cells through an experimental sensory port. Only anatomical DAN->MBON07 pairs
carry a dopamine channel. The selected SMP353 output drives a passive MuJoCo
hinge through a fixed muscle filter. That last mapping replaces missing motor
circuitry and is NOT a reconstructed fly behavior or an action policy.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np

from ..connectome import Subgraph, sha256
from ..paula import Dynamics, build_paula
from ...active_inference.components.body.loaded_hinge import LoadedHinge, DT, MOTOR_GEAR
from ...active_inference.components.body.energy_budget import EnergyBudget
from ...active_inference.core.runtime_checkpoint import save_checkpoint


SOMA_FIELDS = ("S", "O", "r", "b", "t_ref", "F_avg")
PHYSICAL_FIELDS = ("angle", "velocity", "muscle", "delivered_j", "digested_j",
                   "energy_j", "gut_j", "unmet_j")


def groups(graph):
    return {name: [r for r in graph.selected
                   if graph.nodes[r]["annotation"]["hemibrain_type"] == name]
            for name in ("MBON07", "PAM11", "SMP353", "SMP108", "APL")}


def build(graph, rule="native", dopamine=True):
    if rule not in {"native", "dopamine_hebb"}:
        raise ValueError("Unknown existing-rule comparison")
    dynamics = Dynamics()
    prep = build_paula(graph, dynamics)
    roles = groups(graph)
    mbons = {prep.root_to_id[r] for r in roles["MBON07"]}
    dans = {prep.root_to_id[r] for r in roles["PAM11"]}
    kcs = {prep.root_to_id[r] for r in graph.selected
           if graph.nodes[r]["annotation"]["cell_class"] == "Kenyon_Cell"}
    selected = []
    for row, pre, terminal, post, sid in prep.edge_bindings:
        if pre in dans and post in mbons:
            # Preserve the anatomical pair, signed fast current and reciprocal
            # path. Add a declared transmitter/receptor assignment using native
            # channels; count scale is engineering, not receptor measurement.
            cell = prep.network.network.neurons[int(pre)]
            cell.presynaptic_points[int(terminal)].u_o.mod[1] = 1.
            receptor = prep.network.network.neurons[int(post)].postsynaptic_points[int(sid)]
            receptor.u_i.adapt[1] = abs(receptor.u_i.info) if dopamine else 0.
        if post in mbons and pre in kcs:
            selected.append((int(row), int(pre), int(terminal), int(post), int(sid)))
    for nid in mbons:
        params = prep.network.network.neurons[nid].params
        # Existing NeuronParameters default learning rate, applied to MBONs.
        # Other cells retain the published fly execution settings, all positive.
        params.eta_post = .01
        if rule == "dopamine_hebb":
            params.plasticity_mode = "reward_hebb"
            params.nm_plasticity_kappa = -10.
            params.rh_decay = 1.
    prep.assumptions["learning_comparison"] = {
        "rule": rule, "MBON_eta_post": .01, "dopamine_receptor_enabled": dopamine,
        "dopamine": "M[1], unit release on measured PAM11->MBON07 pairs; receiving adapt=count*0.02; original fast current retained",
        "native": "Unchanged legacy_multiplicative equation and threshold/window modulation",
        "dopamine_hebb": "Existing reward_hebb equation, kappa=-10, rh_decay=1. Dopamine suppresses its positive coincidence term; active-port decay remains. No new equation.",
        "limits": "Soma-wide dopamine at MBON, no KC-terminal compartment model or dopamine receptor fit; inhibitory-input sign changes are measured, not prevented",
    }
    return prep, np.array(selected, dtype=np.int64)


def protocol(paired=True, blocks=6):
    phases = []
    def add(name, ticks, cue="", nutrient=False):
        phases.append(dict(name=name, ticks=ticks, cue=cue, nutrient=nutrient))
    add("settle", 200)
    add("naive_A", 200, "A"); add("naive_gap", 200)
    add("naive_C", 200, "C"); add("naive_recovery", 200)
    for block in range(blocks):
        add(f"train{block}_A", 140, "A")
        add(f"train{block}_A_tail", 60, "A", paired)
        add(f"train{block}_gap", 1000)
        add(f"train{block}_unpaired", 60, "", not paired)
        add(f"train{block}_washout", 1000)
        add(f"train{block}_C", 200, "C")
        add(f"train{block}_recovery", 1000)
    add("retention", 1000)
    add("retained_A", 200, "A"); add("retained_gap", 400)
    add("retained_C", 200, "C"); add("final_recovery", 400)
    return phases


def run(graph_path, output, *, rule="native", paired=True, dopamine=True, blocks=6):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(blocks) is not int or not 1 <= blocks <= 6:
        raise ValueError("Bounded prerequisite requires 1..6 blocks")
    random.seed(11); np.random.seed(11)
    graph = Subgraph.load(Path(graph_path))
    prep, selected = build(graph, rule, dopamine)
    net = prep.network
    roles = groups(graph)
    cells = list(net.network.neurons.values())
    cell_rows = {c.id: i for i, c in enumerate(cells)}
    codes = graph.provenance["controlled_codes"]
    body = LoadedHinge(spring=.15)
    # A longer counterbalancing gap should not create an unrelated fuel failure.
    # Capacity scales the existing physical accounting, not a neural threshold.
    from ...active_inference.components.body.energy_budget import EnergyBudgetParameters
    organs = EnergyBudget(energy_j=90., gut_j=0.,
                          params=EnergyBudgetParameters(capacity_j=120.))
    muscle = 0.
    phases = protocol(paired, blocks)
    ticks = sum(p["ticks"] for p in phases)
    output.mkdir(parents=True)
    # Disk-backed every-tick recording; no history list grows with run duration.
    def array(name, shape):
        return np.lib.format.open_memmap(output / (name + ".npy"), mode="w+", dtype=np.float64, shape=shape)
    soma = array("soma", (ticks, len(cells), len(SOMA_FIELDS)))
    modulation = array("modulation", (ticks, len(cells), 2))
    weights = array("weights", (ticks + 1, len(selected)))
    release = array("release", (ticks + 1, len(selected)))
    physical = array("physical", (ticks, len(PHYSICAL_FIELDS)))
    post = [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])] for e in selected]
    pre = [net.network.neurons[int(e[1])].presynaptic_points[int(e[2])] for e in selected]
    weights[0] = [p.u_i.info for p in post]
    release[0] = [p.u_o.info for p in pre]
    np.savez_compressed(output / "identities.npz", selected=selected,
                        cells=np.array([c.id for c in cells]), roots=np.array(list(prep.root_to_id)))
    save_checkpoint(net, output / "birth.paula", sources=[__file__])
    np.savez_compressed(output / "birth-body.npz", state=body.state(),
                        energy=organs.state(), muscle=muscle)
    manifest = dict(graph=str(Path(graph_path).resolve()), graph_sha256=sha256(Path(graph_path) / "manifest.json"),
        assumptions=prep.assumptions, roles=roles, codes=codes, rule=rule,
        paired=paired, dopamine=dopamine, blocks=blocks, phases=phases,
        protocol_version="spaced-nutrient-v2", energy_parameters=asdict(organs.params),
        soma_fields=SOMA_FIELDS, physical_fields=PHYSICAL_FIELDS, seed=11,
        physical_seconds_per_tick=DT,
        physical_limit="Artificial mapping of a PAULA tick to 4 ms; not measured fly dynamics. Loaded hinge replaces walking.",
        sensory_boundary="Each active controlled KC receives current 40; food accepted by the gut drives each PAM11 experimental port at 40 times accepted/0.008 J",
        motor_boundary="Constant current 1.4 at SMP353, no cue-dependent motor command; muscle EMA=.9*old+.1*actual SMP353 spike; actuator command=muscle",
        benefit="Nutrient pump offers .008 J/tick, accepted by existing EnergyBudget gut. Positive work and activation consume energy. No cue identity reaches the organs.",
        record_limit="All somata and modulators each tick; all selected KC->MBON weights/releases each tick; full neural checkpoints and physical states at birth and phase boundaries. Not all boundary weights recorded each tick.",
        dynamics=asdict(Dynamics()), code_sha256=sha256(Path(__file__)))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start = time.perf_counter(); t = 0; reports = []
    for phase in phases:
        begin = t
        active = set(codes.get(phase["cue"], []))
        for _ in range(phase["ticks"]):
            accepted = organs.ingest(.008) if phase["nutrient"] else 0.
            for roots in codes.values():
                for root in roots:
                    prep.stimulate(root, 40. if root in active else 0.)
            for root in roles["PAM11"]:
                prep.stimulate(root, 40. * accepted / .008)
            for root in roles["SMP353"]:
                prep.stimulate(root, 1.4)
            net.run_tick()
            output_spikes = np.mean([net.network.neurons[prep.root_to_id[r]].O for r in roles["SMP353"]])
            muscle = .9 * muscle + .1 * output_spikes
            old_angle = float(body.data.qpos[0])
            body.step(muscle)
            work = max(0., MOTOR_GEAR * muscle * (float(body.data.qpos[0]) - old_angle))
            digested, _, _, _ = organs.advance(DT, work, muscle * muscle * DT)
            soma[t] = [[getattr(c, name) for name in SOMA_FIELDS] for c in cells]
            modulation[t] = [c.M_vector for c in cells]
            weights[t + 1] = [p.u_i.info for p in post]
            release[t + 1] = [p.u_o.info for p in pre]
            physical[t] = [body.data.qpos[0], body.data.qvel[0], muscle, accepted, digested,
                           organs.energy_j, organs.gut_j, organs.unmet_j]
            if not np.isfinite(soma[t]).all() or not np.isfinite(weights[t + 1]).all():
                raise FloatingPointError("Nonfinite neural state")
            t += 1
        report = dict(name=phase["name"], begin=begin, end=t, spikes={
            role: int(soma[begin:t, [cell_rows[prep.root_to_id[r]] for r in roots], 1].sum())
            for role, roots in roles.items()},
            mean_angle=float(physical[begin:t, 0].mean()),
            max_dopamine=float(modulation[begin:t, :, 1].max()),
            weight_mean=float(weights[t].mean()), energy_j=organs.energy_j,
            seconds=time.perf_counter() - start)
        reports.append(report)
        if phase["name"] in {"naive_recovery", f"train{blocks-1}_recovery", "retention", "final_recovery"}:
            save_checkpoint(net, output / (phase["name"] + ".paula"), sources=[__file__])
            np.savez_compressed(output / (phase["name"] + "-body.npz"),
                                state=body.state(), energy=organs.state(), muscle=muscle)
        for a in (soma, modulation, weights, release, physical):
            a.flush()
        (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return reports


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--rule", choices=("native", "dopamine_hebb"), default="native")
    p.add_argument("--unpaired", action="store_true")
    p.add_argument("--no-dopamine-receptor", action="store_true")
    p.add_argument("--blocks", type=int, default=6)
    a = p.parse_args()
    run(a.graph, a.output, rule=a.rule, paired=not a.unpaired,
        dopamine=not a.no_dopamine_receptor, blocks=a.blocks)
