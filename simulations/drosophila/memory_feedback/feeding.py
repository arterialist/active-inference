"""Continuing memory-guided feeding with the same 77-cell anatomical cut.

No new neural equation or connection. Increase the two output cells' sensitivity
to measured neural inputs using existing thresholds, and provide equal cue-
presence excitation as an explicit lateral-horn isolation boundary. The factor
is selected analytically from single-event transmission, not behavior fitting.
The body reaches a fixed food well; contact, not cue identity, permits ingestion.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np

from .acquisition import groups, protocol, SOMA_FIELDS
from .input_rule import build as memory_build
from ..connectome import Subgraph, sha256
from ...active_inference.components.body.loaded_hinge import LoadedHinge, DT, MOTOR_GEAR
from ...active_inference.components.body.energy_budget import EnergyBudget, EnergyBudgetParameters
from ...active_inference.core.runtime_checkpoint import save_checkpoint


BODY_FIELDS = ("angle", "velocity", "muscle", "command", "pump_j", "well_j",
               "digested_j", "energy_j", "gut_j", "unmet_j")
MOTOR_GAIN = 4.
WELL_ANGLE = .04
DOSE_J = .008


def configure(graph, *, sensitive=True):
    prep, selected = memory_build(graph)
    roles = groups(graph)
    src, dst = roles["SMP353"][0], roles["SMP108"][0]
    edge = graph.internal[(graph.internal[:, 0] == int(src)) & (graph.internal[:, 1] == int(dst))]
    if len(edge) != 1 or edge[0, 5] != 1:
        raise ValueError("Expected the identified excitatory output connection")
    target = prep.network.network.neurons[prep.root_to_id[dst]]
    # One event, after native two-tick dendritic attenuation, raises S by
    # w*decay**delay/lambda. Choose the next power of two above b/pulse.
    pulse = float(edge[0, 6])*.02*.95**2/target.params.lambda_param
    required = target.params.b_base/pulse
    factor = float(2**math.ceil(math.log2(required))) if sensitive else 1.
    for role in ("SMP353", "SMP108"):
        for root in roles[role]:
            cell = prep.network.network.neurons[prep.root_to_id[root]]
            cell.params.r_base /= factor
            cell.params.b_base /= factor
            cell.r = cell.params.r_base; cell.b = cell.params.b_base
    prep.assumptions["output_transfer"] = dict(sensitivity=factor,
        source=src, target=dst, source_row=int(edge[0, 8]), counted_synapses=int(edge[0, 4]),
        single_event_delta_S=pulse, minimum_factor_for_cooldown_threshold=required,
        derivation="Next power of two above b/(count*.02*.95**2/lambda), from one anatomical event before behavioral execution",
        cue_presence_drive=1.4/factor,
        boundary="Equal cue-presence excitation of SMP353; zero in blank periods. This is an experimental input, not imputed lateral-horn spikes.",
        limit="Unfitted output operating-point hypothesis. Native thresholds change; measured connections and existing learning equations do not.")
    return prep, selected, factor


class FeedingBody:
    def __init__(self):
        self.body = LoadedHinge(spring=.15)
        self.organs = EnergyBudget(energy_j=90., gut_j=0., params=EnergyBudgetParameters(capacity_j=120.))
        self.muscle = 0.

    def offer(self, pump, well):
        # Contact measured BEFORE the next action. No cue name or learned
        # value enters this physical transducer.
        pump_j = self.organs.ingest(DOSE_J) if pump else 0.
        well_j = self.organs.ingest(DOSE_J) if well and self.body.data.qpos[0] >= WELL_ANGLE else 0.
        return pump_j, well_j

    def step(self, spike, pump_j, well_j):
        self.muscle = .9*self.muscle+.1*spike
        command = MOTOR_GAIN*self.muscle
        old_angle = float(self.body.data.qpos[0])
        self.body.step(command)
        work = max(0., MOTOR_GEAR*command*(float(self.body.data.qpos[0])-old_angle))
        digested, _, _, _ = self.organs.advance(DT, work, command*command*DT)
        return [self.body.data.qpos[0], self.body.data.qvel[0], self.muscle, command,
                pump_j, well_j, digested, self.organs.energy_j, self.organs.gut_j, self.organs.unmet_j]

    def save(self, path):
        np.savez_compressed(path, state=self.body.state(), energy=self.organs.state(), muscle=self.muscle)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.body.restore(z["state"])
            self.muscle = float(z["muscle"])
            for name, value in zip(self.organs.fields, z["energy"], strict=True):
                setattr(self.organs, name, float(value))


def step(net, ids, roles, codes, cue, organism, *, factor, pump=False, well=False, advance=None):
    pump_j, well_j = organism.offer(pump, well)
    active = set(codes.get(cue, ()))
    for code in codes.values():
        for root in code:
            cell = net.network.neurons[ids[root]]
            net.set_external_input(cell.id, cell.params.num_inputs-1, 40. if root in active else 0.)
    for root in roles["PAM11"]:
        cell = net.network.neurons[ids[root]]
        net.set_external_input(cell.id, cell.params.num_inputs-1, 40.*(pump_j+well_j)/DOSE_J)
    for root in roles["SMP353"]:
        cell = net.network.neurons[ids[root]]
        net.set_external_input(cell.id, cell.params.num_inputs-1, 1.4/factor if active else 0.)
    (advance or net.run_tick)()
    spike = float(np.mean([net.network.neurons[ids[r]].O for r in roles["SMP353"]]))
    return organism.step(spike, pump_j, well_j)


def run(graph_path, output, *, paired=True, sensitive=True):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    random.seed(11); np.random.seed(11)
    graph = Subgraph.load(Path(graph_path))
    prep, selected, factor = configure(graph, sensitive=sensitive)
    net = prep.network; roles = groups(graph); codes = graph.provenance["controlled_codes"]
    organism = FeedingBody(); cells = list(net.network.neurons.values())
    rows = {c.id: i for i, c in enumerate(cells)}
    phases = protocol(paired, blocks=2); ticks = sum(p["ticks"] for p in phases)
    output.mkdir(parents=True)
    def array(name, shape):
        return np.lib.format.open_memmap(output/(name+".npy"), mode="w+", dtype=np.float64, shape=shape)
    soma = array("soma", (ticks, len(cells), len(SOMA_FIELDS)))
    modulation = array("modulation", (ticks, len(cells), 2))
    weights = array("weights", (ticks+1, len(selected)))
    release = array("release", (ticks+1, len(selected)))
    body = array("body", (ticks, len(BODY_FIELDS)))
    post = [net.network.neurons[int(e[3])].postsynaptic_points[int(e[4])] for e in selected]
    pre = [net.network.neurons[int(e[1])].presynaptic_points[int(e[2])] for e in selected]
    weights[0] = [p.u_i.info for p in post]; release[0] = [p.u_o.info for p in pre]
    np.savez_compressed(output/"identities.npz", selected=selected, cells=np.array([c.id for c in cells]),
                        roots=np.array(list(prep.root_to_id)))
    manifest = dict(graph=str(Path(graph_path).resolve()), graph_sha256=sha256(Path(graph_path)/"manifest.json"),
        assumptions=prep.assumptions, roles=roles, codes=codes, paired=paired, factor=factor,
        phases=phases, soma_fields=SOMA_FIELDS, body_fields=BODY_FIELDS, motor_gain=MOTOR_GAIN,
        well_angle=WELL_ANGLE, dose_j=DOSE_J, energy_parameters=asdict(organism.organs.params),
        physical_seconds_per_tick=DT, seed=11, code_sha256=sha256(Path(__file__)),
        protocol="Same spaced direct acquisition. At retained_A onset a fixed food well becomes available for all subsequent cues and gaps. Contact alone permits eating.",
        limit="Engineered one-joint feeding action and controlled sensory boundaries, not natural fly locomotion; pump conditioning precedes contact-dependent tests.")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save_checkpoint(net, output/"birth.paula", sources=[__file__]); organism.save(output/"birth-body.npz")
    t = 0; reports = []; started = time.perf_counter(); well = False
    for phase in phases:
        begin = t
        if phase["name"] == "retained_A":
            well = True
        for _ in range(phase["ticks"]):
            body[t] = step(net, prep.root_to_id, roles, codes, phase["cue"], organism,
                           factor=factor, pump=phase["nutrient"], well=well)
            soma[t] = [[getattr(c, f) for f in SOMA_FIELDS] for c in cells]
            modulation[t] = [c.M_vector for c in cells]
            weights[t+1] = [p.u_i.info for p in post]; release[t+1] = [p.u_o.info for p in pre]
            if any(not np.isfinite(a).all() for a in (soma[t], weights[t+1], body[t])):
                raise FloatingPointError("Nonfinite course state")
            t += 1
        report = dict(name=phase["name"], begin=begin, end=t,
            spikes={role: int(soma[begin:t, [rows[prep.root_to_id[r]] for r in rr], 1].sum()) for role, rr in roles.items()},
            max_angle=float(body[begin:t, 0].max()), pump_j=float(body[begin:t, 4].sum()),
            well_j=float(body[begin:t, 5].sum()), energy_j=organism.organs.energy_j,
            seconds=time.perf_counter()-started)
        reports.append(report)
        if phase["name"] in ("naive_recovery", "train1_recovery", "retention", "final_recovery"):
            save_checkpoint(net, output/(phase["name"]+".paula"), sources=[__file__])
            organism.save(output/(phase["name"]+"-body.npz"))
        for a in (soma, modulation, weights, release, body):
            a.flush()
        (output/"summary.json").write_text(json.dumps(reports, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--unpaired", action="store_true")
    p.add_argument("--reference-sensitivity", action="store_true")
    a = p.parse_args(); run(a.graph, a.output, paired=not a.unpaired, sensitive=not a.reference_sensitivity)
